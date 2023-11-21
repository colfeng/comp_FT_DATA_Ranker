#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import copy
import logging
import warnings
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import numpy as np
import torch
import transformers
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'

warnings.filterwarnings("ignore")

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "en": {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )},
    "zh": {
        "prompt_input": (
            "以下是描述任务的指示，配有提供进一步上下文的输入，编写一个适当的回应完成请求\n\n"
            "### 指示：\n{instruction}\n\n### 输入：\n{input}\n\n### 回应："
        ),
        "prompt_no_input": (
            "以下是描述任务的指示，编写一个适当的回应完成请求\n\n"
            "### 指示：\n{instruction}\n\n### 回应："
        )}
}


@dataclass
class ModelArguments:
    old_model_name_or_path: Optional[str] = field(default="Baichuan2-7B-Base")
    new_model_name_or_path: Optional[str] = field(default="30m_0615_baichuan")
    tokenizer: str = field(default="Baichuan2-7B-Base")


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default="10m_en_refine.jsonl")
    lang: str = field(default="en")
    num_proc: int = field(default=1)


@dataclass
class TrainingArguments: #(transformers.TrainingArguments)
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    save_dir: str = field(default='entropy_10m_en.jsonl')


def print_rank(*args, **kwargs):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args, **kwargs)


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    def _get_resized_lm_head(
            self, old_lm_head, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    ):

        if new_num_tokens is None:
            return old_lm_head
        old_num_tokens, old_lm_head_dim = (
            old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
        )

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = False
        if hasattr(old_lm_head, 'bias'):
            has_new_lm_head_bias = old_lm_head.bias is not None

        new_lm_head = old_lm_head.__class__(*new_lm_head_shape, bias=has_new_lm_head_bias)
        new_lm_head.weight = torch.nn.Parameter(new_lm_head.weight.to(
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype))

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

        return new_lm_head

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    old_lm_head = model.get_output_embeddings()
    from torch import nn
    if old_lm_head is not None and not isinstance(old_lm_head, nn.Linear):
        from types import MethodType

        model._get_resized_lm_head = MethodType(_get_resized_lm_head, model)

    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def preprocess(
        format_dataset: Dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        num_proc: int = 1,
) -> Dict:
    def _tokenize_fn(example):
        """Tokenize example"""
        example["source"] = tokenizer(example["source"], return_tensors="pt", padding="longest",
                                      max_length=tokenizer.model_max_length, truncation=True,
                                      add_special_tokens=False)
        example["target"] = tokenizer(example["target"], return_tensors="pt", padding="longest",
                                      max_length=tokenizer.model_max_length, truncation=True,
                                      add_special_tokens=False)

        source_input_id = source_label = example["source"].input_ids[0]
        target_input_id = target_label = torch.cat(
            [example["target"].input_ids[0], torch.tensor([tokenizer.eos_token_id])])
        input_id = torch.cat([source_input_id, target_input_id])
        label = copy.deepcopy(input_id)
        label[:len(source_input_id)] = IGNORE_INDEX

        example["input_ids"] = input_id
        example["labels"] = label
        example["split_ids"] = len(source_input_id)
        return example

    """Preprocess the data by tokenizing."""
    processed_dataset = format_dataset.map(_tokenize_fn, remove_columns=["source", "target"], num_proc=num_proc)
    processed_dataset.set_format("pt", columns=["input_ids", "labels"], output_all_columns=True)

    return processed_dataset


def format_data(lang: str, dataset: Dataset, num_proc: int = 1):
    prompt_input, prompt_no_input = PROMPT_DICT[lang]["prompt_input"], PROMPT_DICT[lang]["prompt_no_input"]

    def add_prompt(example):
        if "instruction" in example and "output" in example:
            example["target"] = example["output"]
            if example.get("input", "") != "":
                example["source"] = prompt_input.format_map(example)
            else:
                example["source"] = prompt_no_input.format_map(example)
            return example
        else:
            raise RuntimeError(f"{example}")

    return dataset.map(add_prompt, remove_columns=["instruction", "input", "output"], num_proc=num_proc)


def SupervisedDataset(data_args, tokenizer: transformers.PreTrainedTokenizer):
    """Dataset for supervised fine-tuning."""
    lang, data_path, num_proc = data_args.lang, data_args.data_path, data_args.num_proc

    dataset = load_dataset('json', data_files=data_path, split="train")
    print_rank(f"There are {len(dataset)} training samples in data path")

    print_rank("Formatting inputs...")
    # * add different formats
    format_dataset = format_data(lang, dataset, num_proc=num_proc)

    print_rank("Tokenizing inputs... This may take some time...")
    dataset_out = preprocess(format_dataset, tokenizer, num_proc=num_proc)
    num_tokens = sum(map(lambda x: len(x['input_ids']), dataset_out))
    print_rank(f"Total {len(dataset_out)} samples [{num_tokens / 10 ** 6: .2f}M tokens] in training!")

    return dataset_out

def update_token_id(model, tokenizer):
    # To solve the bug of llama config
    for name in ['bos', 'eos', 'pad', 'unk']:

        token_id_name = '_'.join([name, 'token_id'])
        token_name = '_'.join([name, 'token'])

        token_id = getattr(tokenizer, token_id_name)
        if token_id is None:
            token_str = getattr(tokenizer, token_name)
            token_id = tokenizer.encode(token_str, add_special_tokens=False)[0]

        setattr(tokenizer, token_id_name, token_id)
        setattr(model.config, token_id_name, token_id)

def entropy(embed):
    x = torch.softmax(embed,1)
    return -torch.mean(torch.sum(x*torch.log(x),1))

def write_res(data, file):
    with open(file, "a", encoding="utf-8") as f0:
        line_dict = dict({'old_entropy':float(data[0]),'new_entropy':float(data[1])})
        js = json.dumps(line_dict, ensure_ascii=False)
        f0.write(js + '\n')
        f0.close()


def score_inference():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print_rank(f"Loading old model from {model_args.old_model_name_or_path}")
    old_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.old_model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        use_cache=False
    )

    print_rank(f"Loading new model from {model_args.new_model_name_or_path}")
    new_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.new_model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        use_cache=False
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )

    print_rank("Old Model class:", old_model.__class__)
    print_rank("New Model class:", new_model.__class__)
    print_rank("Tokenizer class:", tokenizer.__class__)

    # 只更改了bos_token和eos_token的文本显示，但是bos_token和eos_token自己会对应到1，2
    special_tokens_dict = dict()
    if not tokenizer.pad_token:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if not tokenizer.eos_token:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if not tokenizer.bos_token:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if not tokenizer.unk_token:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if special_tokens_dict:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=old_model,
        )

        update_token_id(old_model, tokenizer)

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=new_model,
        )

        update_token_id(new_model, tokenizer)

    data_module = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    with torch.no_grad():
        print_rank("Begin Calculate Score...")
        old_model.cuda(0)
        new_model.cuda(1)
        for sample in tqdm(data_module):
            split = sample['split_ids']

            torch.cuda.empty_cache()

            in_0 = sample['input_ids'][:split].unsqueeze(0).to(old_model.device)
            old_embed = old_model(in_0).logits[0].cpu()
            old_entropy = entropy(old_embed)

            in_1 = sample['input_ids'][:split].unsqueeze(0).to(new_model.device)
            new_embed = new_model(in_1).logits[0].cpu()
            new_entropy = entropy(new_embed)

            write_res((old_entropy, new_entropy), training_args.save_dir)
    print_rank("Finish Inference...")


if __name__ == "__main__":

    score_inference()

