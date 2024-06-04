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

mask_train = train_df['english'].apply(lambda x: not bool(chinese_char_pattern.search(x)))
mask_valid = train_df['english'].apply(lambda x: not bool(chinese_char_pattern.search(x)))

filtered_train_df = train_df[mask_train]
filtered_valid_df = valid_df[mask_valid]

dataset_valid = datasets.Dataset.from_pandas(filtered_valid_df)
dataset_train = datasets.Dataset.from_pandas(filtered_train_df)

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

# data_type = "torch"
# format_columns = ["en_ids", "cn_ids"]
# train_data = train_data.with_format(
#     type=data_type, columns=format_columns, output_all_columns=True
# )

train_data.remove_columns('chinese')
train_data.remove_columns('english')

# valid_data = valid_data.with_format(
#     type=data_type,
#     columns=format_columns,
#     output_all_columns=True,
# )


valid_data.remove_columns('chinese')
valid_data.remove_columns('english')

# test_data = test_data.with_format(
#     type=data_type,
#     columns=format_columns,
#     output_all_columns=True,
# )


test_data.remove_columns('chinese')
test_data.remove_columns('english')
# with open("./dataset/zh_vocab_itos5.pickle", "wb") as f:
#     pickle.dump(json.dumps(zh_vocab.get_itos()), f)
#
# with open("./dataset/zh_vocab_stoi5.pickle", "wb") as f:
#     pickle.dump(json.dumps(zh_vocab.get_stoi()), f)
#
#
# with open("./dataset/en_vocab_itos5.pickle", "wb") as f:
#     pickle.dump(json.dumps(en_vocab.get_itos()), f)
#
# with open("./dataset/en_vocab_stoi5.pickle", "wb") as f:
#     pickle.dump(json.dumps(en_vocab.get_stoi()), f)

with open("./dataset/train_data5.pickle", "wb") as f:
    td = json.dumps({
        'en_tokens': train_data['en_tokens'],
        'cn_tokens': train_data['cn_tokens'],
        'en_ids': train_data['en_ids'],
        'cn_ids': train_data['cn_ids'],
    })
    pickle.dump(td, f)

with open("./dataset/valid_data5.pickle", "wb") as f:
    vd = json.dumps({
        'en_tokens': valid_data['en_tokens'],
        'cn_tokens': valid_data['cn_tokens'],
        'en_ids': valid_data['en_ids'],
        'cn_ids': valid_data['cn_ids'],
    })
    pickle.dump(vd, f)

with open("./dataset/test_data5.pickle", "wb") as f:
    ted = json.dumps({
        'en_tokens': test_data['en_tokens'],
        'cn_tokens': test_data['cn_tokens'],
        'en_ids': test_data['en_ids'],
        'cn_ids': test_data['cn_ids'],
    })
    pickle.dump(ted, f)