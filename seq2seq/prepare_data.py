import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
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

dataset_train = datasets.load_dataset('json', data_files='./ds/translation2019zh_train.json')
dataset_valid = datasets.load_dataset('json', data_files='./ds/translation2019zh_valid.json')

dataset_train = dataset_train['train']
dataset_valid = dataset_valid['train']


en_nlp = spacy.load("en_core_web_sm")
zh_nlp = spacy.load("zh_core_web_sm")

def tokenize_example(example, en_nlp, zh_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example["english"])][:max_length]
    cn_tokens = [token.text for token in zh_nlp.tokenizer(example["chinese"])][:max_length]
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
    "zh_nlp": zh_nlp,
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

min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]
en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["english"],
    min_freq=min_freq,
    specials=special_tokens,
)

zh_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["chinese"],
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

with open("./dataset/zh_vocab.pickle", "wb") as f:
    pickle.dump(zh_vocab, f)

with open("./dataset/en_vocab.pickle", "wb") as f:
    pickle.dump(en_vocab, f)

with open("./dataset/train_data.pickle", "wb") as f:
    pickle.dump(train_data, f)

with open("./dataset/valid_data.pickle", "wb") as f:
    pickle.dump(valid_data, f)

with open("./dataset/test_data.pickle", "wb") as f:
    pickle.dump(test_data, f)