import torch
import datasets
import pandas as pd


# train_dataset = datasets.load_from_disk('./dss/train')
#
# print(train_dataset['en_ids'])

vocab = torch.load('./dsss/en_vocab.pickle')

print(len(vocab.get_itos()))