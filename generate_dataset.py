import os
import pandas as pd
from glob import glob
path1 = "/home/lsh/PycharmProjects/Audio/data/Male speech, man speaking"
path2 = "/home/lsh/PycharmProjects/Audio/data/Outside, rural or natural"
path3 = "/home/lsh/PycharmProjects/Audio/data/snoring"
path4 = "/home/lsh/PycharmProjects/Audio/data/Traffic noise, roadway noise"
path5 = "/home/lsh/PycharmProjects/Audio/data/Vehicle"

paths = [path1, path2, path3, path4, path5]

df = pd.DataFrame()

file_names = []
labels = []
for path in paths:
    file_names += os.listdir(path)
    if path == path3:
        labels += [3] * len(os.listdir(path))
    elif path == path1:
        labels += [1] * len(os.listdir(path))
    elif path == path2:
        labels += [2] * len(os.listdir(path))
    elif path == path4:
        labels += [4] * len(os.listdir(path))
    elif path == path5:
        labels += [5] * len(os.listdir(path))

df["file_name"] = file_names
df["labels"] = labels
print(len(file_names), len(labels))
print(df.head())
df.to_csv("audio.csv", index=False)
df = pd.read_csv("audio.csv")
print(df.head())