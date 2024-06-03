import pickle
with open("./dataset/train_data.pickle", "rb") as f:
    d = pickle.load(f)

print(d)
