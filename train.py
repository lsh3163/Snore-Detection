import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam
import torchaudio
from torch.utils.data import Dataset, DataLoader
import librosa
from fastprogress import master_bar, progress_bar
import numpy as np
import time
from torchvision.models import *
from torchvision.transforms import transforms
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
from sklearn.metrics import classification_report
warnings.filterwarnings(action="ignore")

path1 = "/home/lsh/PycharmProjects/Audio/data/Male speech, man speaking"
path2 = "/home/lsh/PycharmProjects/Audio/data/Outside, rural or natural"
path3 = "/home/lsh/PycharmProjects/Audio/data/snoring"
path4 = "/home/lsh/PycharmProjects/Audio/data/Traffic noise, roadway noise"
path5 = "/home/lsh/PycharmProjects/Audio/data/Vehicle"

num_classes = 10
lr = 3e-3
eta_min = 1e-5
t_max = 10
num_epochs = 20

df = pd.read_csv("audio.csv")
# df = df[df["labels"]!=2]
# df = df[df["labels"]!=4]
train_df, test_df = train_test_split(df, test_size=0.4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_df = train_df.reset_index()
test_df = test_df.reset_index()
print(train_df["labels"].value_counts())


def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    wav,sr = librosa.load(file_path,sr=sr)
    if wav.shape[0]<5*sr:
        wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
    else:
        wav=wav[:5*sr]
    spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    # spec_min, spec_max = spec_norm.min(), spec_norm.max()
    # spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
    # spec_scaled = spec_scaled.astype(np.uint8)
    return spec_norm


# Custom data generator


class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print(self.df["file_name"][idx])

        label = self.df["labels"][idx]
        if label == 1:
            audio_path = os.path.join(path1, self.df["file_name"][idx])
        elif label ==2:
            audio_path = os.path.join(path2, self.df["file_name"][idx])
        elif label == 3:
            audio_path = os.path.join(path3, self.df["file_name"][idx])
        elif label == 4:
            audio_path = os.path.join(path4, self.df["file_name"][idx])
        elif label == 5:
            audio_path = os.path.join(path5, self.df["file_name"][idx])

        image = get_melspectrogram_db(audio_path)
        image = spec_to_image(image)
        x = []
        x.append(image)
        x = np.array(x)
        image = x.astype("float64")

        # print(image.shape, label)
        image = torch.tensor(image, device=device).float()
        label = torch.tensor(label, device=device).long()

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label


model = resnet18(pretrained=True)
model.fc = nn.Linear(512,num_classes)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model = model.to(device)

aug_train = transforms.Compose([])
aug_test = transforms.Compose([])


dataset_train = TrainDataset(df=train_df,
                            img_dir="./",
                            transforms=aug_train)

dataset_test = TrainDataset(df=test_df,
                            img_dir="./",
                            transforms=aug_test)

train_loader = DataLoader(dataset=dataset_train, batch_size=128, shuffle=False)
test_loader = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

print(len(train_loader), len(test_loader))

# model = Classifier(num_classes=num_classes).cuda()
# criterion = nn.BCEWithLogitsLoss().cuda()
# criterion = nn.NLLLoss().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = Adam(params=model.parameters(), lr=lr, amsgrad=False)
scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    avg_loss = 0.
    print("Epoch : {} ".format(epoch))
    print("Train")
    for i, (x_batch, y_batch) in enumerate(train_loader):

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() / len(train_loader)
        if i % 1 == 0:
            _, predicted = torch.max(preds.data, 1)
            # print(y_batch.cpu().numpy(), predicted.cpu().numpy())
            print(i)

    torch.save(model.state_dict(), "model.pth")

    print("Test")
    model.eval()
    valid_preds = np.zeros((len(test_loader), num_classes))
    avg_val_loss = 0.

    real = []
    pred = []
    for i, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(x_batch)
        _, predicted = torch.max(preds.data, 1)

        if i % 100 == 0:
            print(y_batch.cpu().numpy()[0], predicted.cpu().numpy()[0])

        real.append(y_batch.cpu().numpy()[0])
        pred.append(predicted.cpu().numpy()[0])

    print(classification_report(real, pred))
    scheduler.step()

