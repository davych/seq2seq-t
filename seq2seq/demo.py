import pickle
# with open("./dataset/en_vocab.pickle", "rb") as f:
#     en_vocab = pickle.load(f)
#
# with open("./ds_snap/en_vocab.pickle", "wb") as f:
#     pickle.dump("  ".join(en_vocab.get_itos()), f)
#

# with open("./dataset/zh_vocab.pickle", "rb") as f:
#     zh_vocab = pickle.load(f)
#
# with open("./ds_snap/zh_vocab.pickle", "wb") as f:
#     pickle.dump("  ".join(zh_vocab.get_itos()), f)
import json

import datasets
# import pandas as pd

with open("./dataset/train_data4.pickle", "rb") as f:
    train_data = pickle.load(f)
dt = json.loads(train_data)

# # Convert DataFrame to Hugging Face Dataset
hf_dataset = datasets.Dataset.from_dict(dt)
#
data_type = "torch"
format_columns = ["en_ids", "cn_ids"]
train_data = hf_dataset.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)

print(len(train_data), train_data, train_data[0])
# 将tensor转换为列表
# for key in ['en_ids', 'cn_ids']:
#     train_data[key] = train_data[key].tolist()

# arr = []
#
# for i, item in enumerate(train_data):
#     print(i)
#     item['en_ids'] = item['en_ids'].tolist()
#     item['cn_ids'] = item['cn_ids'].tolist()
#     arr.append(item)
#
# # 将字典转换为json字符串
# json_str = json.dumps(arr)
# # print(train_data[0]['cn_ids'].tolist())
#
# with open("./ds_snap/train_data.pickle", "wb") as f:
#     pickle.dump(json_str, f)