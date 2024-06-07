import torch
import datasets
import torchtext
import pandas as pd


# train_dataset = datasets.load_from_disk('./dss/train')
#
# print(train_dataset['en_ids'])

# vocab = torch.load('./ns/en_vocab.pickle')
#
# print(len(vocab.get_itos()), vocab.get_itos())
train_dataset = datasets.load_from_disk('./nss/train')

min_freq = 80
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

zh_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_dataset["cn_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

print(len(zh_vocab.get_itos()))