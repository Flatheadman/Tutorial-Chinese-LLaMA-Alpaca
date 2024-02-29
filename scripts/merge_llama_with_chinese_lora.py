"""
把lora模型的参数经过计算融入到基座模型中，从而无需每次前向传播的时候都做融合计算。
Usage: 
python merge_llama_with_chinese_lora.py \
    --base_model path/to/llama/model \
    --lora_model path/to/first/lora/model [path/to/second/lora/model] \
    --output_type [pth|huggingface] \
    --output_dir path/to/output/dir
"""
import argparse
import json
import os
import gc
import torch
import peft
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, required=True,
                    type=str, help="Please specify a base_model")
parser.add_argument('--lora_model', default=None, required=True,
                    type=str, help="Please specify LoRA models to be merged (ordered); use commas to separate multiple LoRA models.")
parser.add_argument('--offload_dir', default=None, type=str,
                    help="(Optional) Please specify a temp folder for offloading (useful for low-RAM machines). Default None (disable offload).")
parser.add_argument('--output_type', default='pth',choices=['pth','huggingface'], type=str,
                    help="save the merged model in pth or huggingface format.")
parser.add_argument('--output_dir', default='./', type=str)


emb_to_model_size = {
    4096 : '7B',
    5120 : '13B',
    6656 : '33B',
    8192 : '65B',
}
num_shards_of_models = {'7B': 1, '13B': 2, '33B': 4, '65B': 8}
params_of_models = {
    '7B':
        {
        "dim": 4096,
        "multiple_of": 256,
        "n_heads": 32,
        "n_layers": 32,
        "norm_eps": 1e-06,
        "vocab_size": -1,
        },
    '13B':
        {
        "dim": 5120,
        "multiple_of": 256,
        "n_heads": 40,
        "n_layers": 40,
        "norm_eps": 1e-06,
        "vocab_size": -1,
        },
    '33B':
        {
        "dim": 6656,
        "multiple_of": 256,
        "n_heads": 52,
        "n_layers": 60,
        "norm_eps": 1e-06,
        "vocab_size": -1,
        },
    '65B':
        {
        "dim": 8192,
        "multiple_of": 256,
        "n_heads": 64,
        "n_layers": 80,
        "norm_eps": 1e-05,
        "vocab_size": -1,
        },
}

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

# Borrowed and modified from https://github.com/tloen/alpaca-lora
'''
这个Python函数 translate_state_dict_key(k) 的目的是将一种模型的字典键转换为另一种模型的字典键。这在你需要将预训练的权重加载到具有不同架构的模型时非常有用。
函数接收一个参数 k，这是需要被转换的键。
函数首先使用 replace 方法从键 k 中移除前缀 "base_model.model."。
然后，函数通过一系列的 if 和 elif 语句，检查键 k 的值，并将其转换为新的键。例如，如果 k 的值为 "model.embed_tokens.weight"，则函数返回 "tok_embeddings.weight"。
'''
def translate_state_dict_key(k):
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError


'''
这个Python函数 `unpermute(w)` 的目的是对输入的张量 `w` 进行一系列的变换。
函数接收一个参数 `w`，这是需要被变换的张量。
首先，函数使用 `view` 方法将 `w` 重塑为一个四维张量，其形状为 `(n_heads, 2, dim // n_heads // 2, dim)`。这里，`n_heads` 和 `dim` 是全局变量，分别表示注意力头的数量和维度。
然后，函数使用 `transpose` 方法交换第二和第三维度。
最后，函数使用 `reshape` 方法将张量重塑为 `(dim, dim)` 形状。
总的来说，这个函数的目的是对输入的张量进行一系列的重塑和转置操作，以达到特定的数据组织方式。
'''
def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)
    )

'''
这段Python代码是用于保存模型的状态字典（state dictionary）的分片（shards）。这个函数的主要目的是将模型的状态字典分解成多个部分（如果需要的话），然后将这些部分保存到磁盘上。

函数的主要步骤如下：

1. 使用`torch.no_grad()`上下文管理器，这是为了在保存模型时不计算梯度，以节省内存。

2. 检查`num_shards`的值。如果它等于1，那么就不需要分片，直接将状态字典保存到文件中。如果它大于1，那么就需要将状态字典分解成多个部分。

3. 对于需要分片的情况，函数首先创建一个新的状态字典列表`new_state_dicts`，然后遍历原始的状态字典`model_sd`。

4. 对于状态字典中的每个键值对，函数首先调用`translate_state_dict_key(k)`来翻译键名。然后，根据键名的不同，函数采用不同的方式来分解值。

5. 分解后的值被保存到新的状态字典中，然后原始的键值对被删除，以释放内存。

6. 最后，函数将新的状态字典保存到磁盘上，并将模型的参数保存到一个JSON文件中。

这个函数的主要用途是在处理大型模型时，将模型的状态字典分解成多个部分，以便于在有限的内存中处理。
'''
def save_shards(model_sd, num_shards: int):
    # Add the no_grad context manager
    with torch.no_grad():
        # 如果不需要分片，直接变换状态字典的键名和值，并保存到磁盘上
        if num_shards == 1:
            new_state_dict = {}
            for k, v in model_sd.items():
                new_k = translate_state_dict_key(k)
                if new_k is not None:
                    if "wq" in new_k or "wk" in new_k:
                        new_state_dict[new_k] = unpermute(v)    # 对特定状态键对应的状态值张量进行反置换变换
                    else:
                        new_state_dict[new_k] = v

            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving shard 1 of {num_shards} into {output_dir}/consolidated.00.pth")
            torch.save(new_state_dict, output_dir + "/consolidated.00.pth")
            with open(output_dir + "/params.json", "w") as f:
                json.dump(params, f)
        # 如果需要分片，首先创建一个新的状态字典列表，然后遍历原始的状态字典
        else:
            new_state_dicts = [dict() for _ in range(num_shards)]
            for k in list(model_sd.keys()):
                v = model_sd[k]
                new_k = translate_state_dict_key(k)
                if new_k is not None:
                    if new_k=='tok_embeddings.weight':
                        print(f"Processing {new_k}")
                        assert v.size(1)%num_shards==0
                        splits = v.split(v.size(1)//num_shards,dim=1)
                    elif new_k=='output.weight':
                        print(f"Processing {new_k}")
                        if v.size(0)%num_shards==0:
                            splits = v.split(v.size(0)//num_shards,dim=0)
                        else:
                            size_list = [v.size(0)//num_shards] * num_shards
                            size_list[-1] += v.size(0)%num_shards
                            splits = v.split(size_list, dim=0) # 13B: size_list == [24976,24977]
                    elif new_k=='norm.weight':
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards
                    elif 'ffn_norm.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards
                    elif 'attention_norm.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = [v] * num_shards


                    elif 'w1.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0)//num_shards,dim=0)
                    elif 'w2.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(1)//num_shards,dim=1)
                    elif 'w3.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0)//num_shards,dim=0)


                    elif 'wo.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(1)//num_shards,dim=1)

                    elif 'wv.weight' in new_k:
                        print(f"Processing {new_k}")
                        splits = v.split(v.size(0)//num_shards,dim=0)

                    elif "wq.weight" in new_k or "wk.weight" in new_k:
                        print(f"Processing {new_k}")
                        v = unpermute(v)
                        splits = v.split(v.size(0)//num_shards,dim=0)
                    else:
                        print(f"Unexpected key {new_k}")
                        raise ValueError
                    for sd,split in zip(new_state_dicts,splits):
                        sd[new_k] = split.clone()
                        del split
                    del splits
                del model_sd[k],v
                gc.collect()    # Effectively enforce garbage collection

            os.makedirs(output_dir, exist_ok=True)
            for i,new_state_dict in enumerate(new_state_dicts):
                print(f"Saving shard {i+1} of {num_shards} into {output_dir}/consolidated.0{i}.pth")
                torch.save(new_state_dict, output_dir + f"/consolidated.0{i}.pth")
            with open(output_dir + "/params.json", "w") as f:
                print(f"Saving params.json into {output_dir}/params.json")
                json.dump(params, f)


if __name__=='__main__':

    args = parser.parse_args()
    base_model_path = args.base_model
    lora_model_paths = [s.strip() for s in args.lora_model.split(',') if len(s.strip())!=0] # 只保留不为空的路径
    output_dir = args.output_dir
    output_type = args.output_type
    offload_dir = args.offload_dir

    print(f"Base model: {base_model_path}")
    print(f"LoRA model(s) {lora_model_paths}:")

    '''
    离线加载模型：
        在PyTorch中，离线加载模型主要是指将模型的参数（也就是状态字典）保存到磁盘上，然后在需要的时候从磁盘上加载这些参数，而不是直接将参数保存在内存中。
    这种方式对于内存较小的机器非常有用，因为它可以减少内存的使用。但是，这种方式的加载速度可能会比直接从内存中加载慢一些。
    '''
    if offload_dir is not None:
        # Load with offloading, which is useful for low-RAM machines.
        # Note that if you have enough RAM, please use original method instead, as it is faster.
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            offload_folder=offload_dir,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
    else:
        # 直接加载到内存中
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
        )

    ## 从文本嵌入层的维度推断模型大小，如：4096 -> 7B， 5120 -> 13B
    embedding_size = base_model.get_input_embeddings().weight.size(1)
    model_size = emb_to_model_size[embedding_size]
    print(f"Peft version: {peft.__version__}")
    print(f"Loading LoRA for {model_size} model")

    lora_model = None
    lora_model_sd = None
    for lora_index, lora_model_path in enumerate(lora_model_paths): 
        print(f"Loading LoRA {lora_model_path}...")
        # 获取基座和LoRA模型的词汇表长度
        tokenizer = LlamaTokenizer.from_pretrained(lora_model_path)
        print(f"base_model vocab size: {base_model.get_input_embeddings().weight.size(0)}")
        print(f"tokenizer vocab size: {len(tokenizer)}")

        
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        # 如果lora词汇表长度小于基座词汇表长度，抛出异常
        assert len(tokenizer) >= model_vocab_size, \
        (f"The vocab size of the tokenizer {len(tokenizer)} is smaller than the vocab size of the base model {model_vocab_size}\n"
        "This is not the intended use. Please check your model and tokenizer.")
        # 如果lora词汇表长度大于基座词汇表长度，扩展基座词汇表长度
        if model_vocab_size != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"Extended vocabulary size to {len(tokenizer)}")

        # 获取基座模型中第一层自注意力（self-attention）的查询（query）投影的权重，并创建这个权重的一个副本。
        first_weight = base_model.model.layers[0].self_attn.q_proj.weight
        first_weight_old = first_weight.clone()

        print(f"Loading LoRA weights")
        # 检查peft.LoraModel是否有merge_and_unload这个属性或方法。如果有，就使用merge_and_unload方法；否则，就使用自定义的方法。
        if hasattr(peft.LoraModel,'merge_and_unload'):
            try:
                lora_model = PeftModel.from_pretrained(
                    base_model,
                    lora_model_path,
                    device_map={"": "cpu"},
                    torch_dtype=torch.float16,
                )
            except RuntimeError as e:
                if '[49953, 4096]' in str(e):
                    print("The vocab size of the tokenizer does not match the vocab size of the LoRA weight. \n"
                           "Did you misuse the LLaMA tokenizer with the Alpaca-LoRA weight?\n"
                           "Make sure that you use LLaMA tokenizer with the LLaMA-LoRA weight and Alpaca tokenizer with the Alpaca-LoRA weight!")
                raise e
            assert torch.allclose(first_weight_old, first_weight)
            print(f"Merging with merge_and_unload...")
            # merge_and_unload方法是将LoRA模型（lora_model）与基础模型（base_model）进行合并，并卸载LoRA模型的参数。这个方法的目的是将LoRA模型的参数融入到基础模型中，然后释放LoRA模型的内存。
            base_model = lora_model.merge_and_unload()
        else:
            # 获取基座模型和lora模型的状态字典
            base_model_sd = base_model.state_dict()
            try:
                lora_model_sd = torch.load(os.path.join(lora_model_path,'adapter_model.bin'),map_location='cpu')
            except FileNotFoundError:
                print("Cannot find lora model on the disk. Downloading lora model from hub...")
                filename = hf_hub_download(repo_id=lora_model_path,filename='adapter_model.bin')
                lora_model_sd = torch.load(filename,map_location='cpu')
            if 'base_model.model.model.embed_tokens.weight' in lora_model_sd:
                assert lora_model_sd['base_model.model.model.embed_tokens.weight'].shape[0]==len(tokenizer), \
                ("The vocab size of the tokenizer does not match the vocab size of the LoRA weight. \n"
                "Did you misuse the LLaMA tokenizer with the Alpaca-LoRA weight?\n"
                "Make sure that you use LLaMA tokenizer with the LLaMA-LoRA weight and Alpaca tokenizer with the Alpaca-LoRA weight!")

            lora_config = peft.LoraConfig.from_pretrained(lora_model_path)
            lora_scaling = lora_config.lora_alpha / lora_config.r
            fan_in_fan_out = lora_config.fan_in_fan_out
            lora_keys = [k for k in lora_model_sd if 'lora_A' in k]
            non_lora_keys = [k for k in lora_model_sd if not 'lora_' in k]

            '''
            这段Python代码是遍历non_lora_keys列表，并将lora_model_sd字典中对应的值复制到base_model_sd字典中对应的键值。

            具体来说：
            for k in non_lora_keys:是遍历non_lora_keys列表，k是列表中的每一个元素。
            print(f"merging {k}")是打印出正在合并的键值。
            original_k = k.replace('base_model.model.','')是将k中的'base_model.model.'替换为''，并将结果赋值给original_k。这是为了获取原始模型中对应的键值。
            base_model_sd[original_k].copy_(lora_model_sd[k])是将lora_model_sd字典中k对应的值复制到base_model_sd字典中original_k对应的键值。copy_方法是PyTorch中的一个方法，用于将一个张量的值复制到另一个张量
            '''
            for k in non_lora_keys:
                print(f"merging {k}")
                original_k = k.replace('base_model.model.','')
                base_model_sd[original_k].copy_(lora_model_sd[k])   # 张量的原地操作，直接改变原张量的值

            # 遍历lora keys列表，使用数学算法将lora模型的参数融入到基座模型中（后续不用每次都做融合计算）
            for k in lora_keys:
                print(f"merging {k}")
                original_key = k.replace('.lora_A','').replace('base_model.model.','')
                assert original_key in base_model_sd
                lora_a_key = k
                lora_b_key = k.replace('lora_A','lora_B')
                base_model_sd[original_key] += (
                    transpose(lora_model_sd[lora_b_key].float() @ lora_model_sd[lora_a_key].float(),fan_in_fan_out) * lora_scaling
                )
                assert base_model_sd[original_key].dtype == torch.float16

        # did we do anything?
        assert not torch.allclose(first_weight_old, first_weight)

    # 把分词器保存到本地磁盘
    tokenizer.save_pretrained(output_dir)

    # 把合并后的模型保存到本地磁盘，支持两种格式：pth和huggingface
    if output_type=='huggingface':
        print("Saving to Hugging Face format...")
        LlamaForCausalLM.save_pretrained(base_model, output_dir) #, state_dict=deloreanized_sd)
    # 保存成pth格式时，需要处理分片保存逻辑
    else: # output_type=='pth
        print("Saving to pth format...")

        base_model_sd = base_model.state_dict()
        del lora_model, base_model, lora_model_sd

        params = params_of_models[model_size]
        num_shards = num_shards_of_models[model_size]
        n_layers = params["n_layers"]
        n_heads = params["n_heads"]
        dim = params["dim"]
        dims_per_head = dim // n_heads
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

        save_shards(model_sd=base_model_sd, num_shards=num_shards)
