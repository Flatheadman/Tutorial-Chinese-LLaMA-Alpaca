#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import numpy as np
import math
import os
import sys
# dataclass是Python 3.7中引入的一个装饰器，用于自动添加特殊方法（如__init__和__repr__）到类中。field函数则用于定制dataclass类的字段。
from dataclasses import dataclass, field
# itertools模块包含创建有效循环的函数，chain函数可以把一组迭代对象串联起来，形成一个更大的迭代器。
from itertools import chain
# 这行代码从typing模块中导入了Optional, List, Dict, Any和Mapping。这些都是Python的类型注解，用于在代码中标注变量、函数返回值、函数参数等的预期类型，以提高代码的可读性和可维护性。Optional：用于标注变量或返回值
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import datasets
from datasets import load_dataset, concatenate_datasets
import torch

import transformers
from transformers import (
    CONFIG_MAPPING, # 这是一个字典，用于映射不同预训练模型的配置类。例如，你可以使用CONFIG_MAPPING["gpt2"]来获取GPT-2模型的配置。
    MODEL_FOR_CAUSAL_LM_MAPPING, # 这是一个字典，用于映射不同预训练模型的模型类。例如，你可以使用MODEL_FOR_CAUSAL_LM_MAPPING["gpt2"]来获取GPT-2模型的模型类。
    AutoConfig, # 这是一个自动选择模型配置的类。你可以根据模型名称自动获取相应的配置。
    AutoModelForCausalLM, # 这是一个自动选择模型的类。你可以根据模型名称自动获取相应的模型。
    LlamaForCausalLM, # 这是一个自定义的Causal Language Modeling模型，可能是你自己定义的。
    LlamaTokenizer, # 这是一个自定义的tokenizer，可能是你自己定义的。
    AutoTokenizer, # 这是一个自动选择tokenizer的类。你可以根据模型名称自动获取相应的tokenizer。
    HfArgumentParser, # 这是一个用于解析命令行参数的类。它可以自动将命令行参数转换为Python对象。
    Trainer, # 这是一个用于训练模型的类。它封装了训练模型的所有细节，包括数据加载、优化器、学习率调度器等。
    TrainingArguments, # 这是一个用于训练模型的参数类。它包含了训练模型时的所有参数，如学习率、批大小、训练轮数等。
    is_torch_tpu_available, # 这是一个函数，用于检查当前环境是否支持TPU。
    set_seed, # 这是一个函数，用于设置随机种子，以便实现可重复的随机结果。
)
from transformers.testing_utils import CaptureLogger # 这是一个用于捕获日志的类。它可以捕获日志输出，以便在测试中进行断言。
from transformers.trainer_utils import get_last_checkpoint # 这是一个函数，用于获取最后一个检查点的路径。
from transformers.utils import send_example_telemetry   # 这是一个函数，用于发送遥测数据。遥测数据是用于收集用户使用模型的信息，以便更好地分配资源来维护模型。
from transformers.utils.versions import require_version # 这是一个函数，用于检查当前环境是否安装了指定的库。如果没有安装，它会抛出一个异常。

from sklearn.metrics import accuracy_score # 这是一个函数，用于计算准确率。
from peft import (
    LoraConfig,  # 这是一个类，用于配置Lora模型。
    TaskType,  # 这是一个枚举类，用于指定任务类型。
    get_peft_model,  # 这是一个函数，用于获取PEFT模型。
    PeftModel,  # 这是一个类，用于定义PEFT模型。
    get_peft_model_state_dict # 这是一个函数，用于获取PEFT模型的状态字典。
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR # 这是一个字符串，用于指定检查点目录的前缀。

'''
这段代码定义了一个自定义的回调类SavePeftModelCallback，它继承自transformers.TrainerCallback，用于在训练过程中保存模型和分词器。

save_model方法：这个方法用于保存模型和分词器到指定的文件夹。它接受三个参数：args表示训练参数，state表示当前训练状态，kwargs表示其他额外的参数。
首先，它检查当前状态中是否存在最佳模型检查点，如果存在，则将模型保存在该检查点文件夹下，否则将模型保存在输出目录中。然后，构造模型保存路径peft_model_path，
并调用模型和分词器的save_pretrained方法保存模型和分词器到该路径下。

on_save方法：这个方法是transformers.Trainer在保存检查点时调用的钩子方法。它接受相同的参数，并在保存检查点时调用save_model方法保存模型和分词器，
并返回control参数的值，通常是control.nochange，表示不对训练过程做任何修改。

on_train_end方法：这个方法是transformers.Trainer在训练结束时调用的钩子方法。它也接受相同的参数，并在训练结束时调用save_model方法保存最终的模型和分词器到输出目录中。

这个自定义的回调类的作用是在训练过程中保存模型和分词器，确保在训练结束时或在保存检查点时都能够保存当前模型的状态。
'''
class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "pt_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    
    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "pt_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)


def accuracy(predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }
'''
这个函数名为`accuracy`，它是用来计算模型预测的准确性的。准确性是衡量模型性能的一种方式，尤其在分类任务中非常常见。下面我们逐部分详细解释：

### 函数定义
```python
def accuracy(predictions, references, normalize=True, sample_weight=None):
```
- `def`是Python中定义函数的关键词。
- `accuracy`是这个函数的名称。
- 括号中的`predictions, references, normalize=True, sample_weight=None`是函数接收的参数：
  - `predictions`是模型预测的结果，通常是一个列表或数组，包含了对测试数据的预测类别。
  - `references`是实际的正确结果，也就是每个测试样本的真实类别，用来和`predictions`进行比较。
  - `normalize=True`是一个可选参数，表示是否将准确率标准化（或说归一化），即转换为0到1之间的值。如果为False，则返回正确预测的数量。
  - `sample_weight=None`也是一个可选参数，允许对每个样本指定不同的权重，这在处理不平衡数据集时特别有用。

### 函数主体
```python
    return {
        "accuracy": float(
            accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
        )
    }
```
- `return`关键字表示函数的输出。
- 函数返回一个字典，其中包含了一个键值对。键是字符串`"accuracy"`，而值是通过调用`accuracy_score`函数计算得到的准确率，转换为浮点数（`float`）。
- `accuracy_score`是一个计算准确率的函数，这里假设它来自于某个库（如`sklearn.metrics`），它比较`references`（真实值）和`predictions`（预测值），根据`normalize`和`sample_weight`参数来计算准确率。
  - 如果`normalize`为True，`accuracy_score`返回的是正确预测的比例（即准确率），值在0到1之间。
  - 如果`normalize`为False，它返回正确预测的数量。
  - `sample_weight`参数允许为每个样本分配不同的权重，在计算准确率时考虑这些权重，这对于某些样本比其他样本更重要的情况很有用。

### 专业知识点解释
- **准确率(Accuracy)**：是最直观的性能指标，它是正确预测的数量除以总预测数量。准确率是一个很好的度量标准，当且仅当各类别样本数量大致相等时。在不平衡的数据集中，它可能会给出误导性的高值。
- **归一化(Normalization)**：在这个上下文中，归一化指的是将准确率转换为0到1之间的值，使结果易于理解和比较。
- **样本权重(Sample Weight)**：在某些情况下，不是所有的样本都 equally 重要，某些样本可能比其他样本更加关键。通过为每个样本分配权重，我们可以让模型在计算准确率时更加注重这些重要的样本。

通过这个`accuracy`函数，我们能够以字典的形式获取模型预测的准确性指标，这对于评估和比较模型的性能非常有用。
'''

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)
'''
这段代码定义了一个名为`compute_metrics`的函数，它的作用是计算模型在评估（或验证）阶段的性能指标。这里主要关注的性能指标依然是准确率。下面我们逐行解释它的功能：

### 函数定义
```python
def compute_metrics(eval_preds):
```
这行代码定义了一个名为`compute_metrics`的函数，它接收一个参数`eval_preds`，这个参数通常是一个包含两个部分的元组（tuple），第一部分是模型的预测结果，第二部分是对应的真实标签。

### 数据处理
```python
    preds, labels = eval_preds
```
这行代码将元组`eval_preds`分解为两个变量`preds`和`labels`，`preds`是模型的预测结果，`labels`是真实标签。

```python
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
```
这两行代码对`labels`和`preds`进行了处理：
- `labels[:, 1:]`取`labels`数组的每一行，从第二个元素到最后一个元素，这个操作可能是为了去掉一些特殊的标签，比如在序列任务中，第一个标签可能是起始符号，不需要考虑在准确率计算中。
- `preds[:, :-1]`取`preds`数组的每一行，从第一个元素到倒数第二个元素，这可能是为了与`labels`对齐，去掉预测序列中最后一个元素，通常是终止符号。
- `.reshape(-1)`将二维数组变形成一维数组，以便计算准确率。`-1`表示自动计算这一维的大小。

### 计算并返回准确率
```python
    return accuracy(predictions=preds, references=labels)
```
这行代码调用了之前定义的`accuracy`函数，用处理过的预测结果`preds`和标签`labels`来计算准确率。

整个`compute_metrics`函数的作用是接收模型的预测结果和真实标签，对它们进行预处理以确保数据格式正确，然后计算并返回模型的准确率。这样我们就可以知道模型在验证阶段的表现如何，从而作出相应的调整和改进。
'''


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError: # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    debug_mode : Optional[bool] = field(default=False)
    peft_path : Optional[str] = field(default=None)


logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)


    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        lm_datasets = []
        path = Path(data_args.dataset_dir)
        files = [file.name for file in path.glob("*.txt")]
        if training_args.debug_mode is True:
            files = [files[0]]
        for idx, file in enumerate(files):
            data_file = os.path.join(path, file)
            filename = ''.join(file.split(".")[:-1])
            cache_path = os.path.join(data_args.data_cache_dir, filename)
            os.makedirs(cache_path, exist_ok=True)
            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'training datasets-{filename} has been loaded from disk')
            except Exception:
                cache_dir = os.path.join(data_args.data_cache_dir, filename+"_text")
                os.makedirs(cache_dir, exist_ok=True)
                raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
                logger.info(f"{file} has been loaded")
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names = {k: os.path.join(cache_dir, 'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names = {k: os.path.join(cache_dir, 'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {block_size}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
            if idx == 0:
                lm_datasets = processed_dataset['train']
            else:
                assert lm_datasets.features.type == processed_dataset["train"].features.type
                lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])

        lm_datasets = lm_datasets.train_test_split(test_size = data_args.validation_split_percentage)

    if training_args.do_train:
        train_dataset = lm_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    if training_args.do_eval:
        eval_dataset = lm_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("training example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))



    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model_vocab_size = model.get_output_embeddings().weight.size(0)
    if not (
       (model_vocab_size==32000 and len(tokenizer)==49953) or \
       (model_vocab_size==32000 and len(tokenizer)==32000) or \
       (model_vocab_size==49953 and len(tokenizer)==49953) or \
       (model_vocab_size==49954 and len(tokenizer)==49954)
    ):
        raise ValueError(
            f"The combination of base model (size: {model_vocab_size}) and tokenizer (size: {len(tokenizer)}) is not a valid configuration. Please check our project wiki for further information. \n"
            "Valid configurations (base model / tokenizer):\n"
            "- Continue pre-training original LLaMA: 32000 / 32000 \n"
            "- Pre-training Chinese LLaMA based on original LLaMA: 32000 / 49953 \n"
            "- Continue pre-training Chinese LLaMA: 49953 / 49953 \n"
            "- Continue pre-training Chinese Alpaca: 49954 / 49954 \n")

    model.resize_token_embeddings(len(tokenizer))
    if training_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, training_args.peft_path)
    else:
        logger.info("Init new peft model")
        target_modules = training_args.trainable.split(',')
        modules_to_save = training_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        lora_rank = training_args.lora_rank
        lora_dropout = training_args.lora_dropout
        lora_alpha = training_args.lora_alpha
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"lora_rank: {lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save)
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )
    trainer.add_callback(SavePeftModelCallback)
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()