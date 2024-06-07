import json
import re

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import spacy
import datasets
import jieba
import torchtext
import tqdm
import evaluate
import os
import pickle


seed = 1234

# 标准化随机生成器的种子
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_arr = []
valid_arr = []

with open('./ds/translation2019zh_train.json', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        data = json.loads(line)
        train_arr.append(data)

with open('./ds/translation2019zh_valid.json', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        data = json.loads(line)
        valid_arr.append(data)

train_df = pd.DataFrame(train_arr)
valid_df = pd.DataFrame(valid_arr)

chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')

mask_train1 = train_df['english'].apply(lambda x: not bool(chinese_char_pattern.search(x)))
mask_valid1 = train_df['english'].apply(lambda x: not bool(chinese_char_pattern.search(x)))

mask_train2 = train_df['chinese'].apply(lambda x: bool(chinese_char_pattern.search(x)))
mask_valid2 = train_df['chinese'].apply(lambda x: bool(chinese_char_pattern.search(x)))

filtered_train_df = train_df[mask_train1 & mask_train2]
filtered_valid_df = valid_df[mask_valid1 & mask_valid2]

dataset_valid = datasets.Dataset.from_pandas(filtered_valid_df)
dataset_train = datasets.Dataset.from_pandas(filtered_train_df)

en_nlp = spacy.load("en_core_web_sm")

def tokenize_example(example, en_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example["english"])][:max_length]
    cn_tokens = list(jieba.cut(example["chinese"]))[:max_length]
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        cn_tokens = [token.lower() for token in cn_tokens]
    en_tokens = [sos_token] + en_tokens + [eos_token]
    cn_tokens = [sos_token] + cn_tokens + [eos_token]
    return {"en_tokens": en_tokens, "cn_tokens": cn_tokens}

max_length = 1_000
lower = True
sos_token = "<sos>"
eos_token = "<eos>"
fn_kwargs = {
    "en_nlp": en_nlp,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}

split_dataset = dataset_train.train_test_split(test_size=0.0001)
dataset_test = split_dataset['test']
dataset_train = split_dataset['train']


train_data = dataset_train.map(tokenize_example, fn_kwargs=fn_kwargs)
valid_data = dataset_valid.map(tokenize_example, fn_kwargs=fn_kwargs)
test_data = dataset_test.map(tokenize_example, fn_kwargs=fn_kwargs)
min_freq = 50
unk_token = "<unk>"
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"
special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]
en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["en_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)
zh_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["cn_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)
assert en_vocab[unk_token] == zh_vocab[unk_token]
assert en_vocab[pad_token] == zh_vocab[pad_token]
unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

en_vocab.set_default_index(unk_index)
zh_vocab.set_default_index(unk_index)

def numericalize_example(example, en_vocab, zh_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    zh_ids = zh_vocab.lookup_indices(example["cn_tokens"])
    return {"en_ids": en_ids, "cn_ids": zh_ids}

fn_kwargs = {"en_vocab": en_vocab, "zh_vocab": zh_vocab}
train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)

data_type = "torch"
format_columns = ["en_ids", "cn_ids"]

train_data = train_data.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)
valid_data = valid_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)
test_data = test_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)
print(len(zh_vocab.get_itos()), len(en_vocab.get_itos()))
torch.save(en_vocab, './nss/en_vocab.pickle')
torch.save(zh_vocab, './nss/zh_vocab.pickle')
train_data.save_to_disk('./nss/train')
valid_data.save_to_disk('./nss/valid')
test_data.save_to_disk('./nss/test')
