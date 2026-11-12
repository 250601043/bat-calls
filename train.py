import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ["Biting", "Feeding", "Fighting", "Mating protest", "Sleeping"]
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

x = torch.from_numpy(np.load("data/x.npy"))
y = pd.read_csv("data/y.csv")


def loader(split):
    rows = torch.tensor((y.split == split).to_numpy())
    labels = torch.tensor(y.label.to_numpy())
    return DataLoader(TensorDataset(x[rows], labels[rows]), batch_size=64, shuffle=split == "train")


def run(images):
    images = images.to(device).float() / 40 + 1
    return model(images[:, None].repeat(1, 3, 1, 1))


def predict(data):
    model.eval()
    truth, prediction = [], []
    with torch.no_grad():
        for images, labels in data:
            truth += labels.tolist()
            prediction += run(images).argmax(1).tolist()
    return np.array(truth), np.array(prediction)


train, val = loader("train"), loader("val")
model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(512, len(CLASSES))
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    for images, labels in train:
        loss = loss_function(run(images), labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    truth, prediction = predict(train)
    print(f"epoch {epoch + 1}: train accuracy {(truth == prediction).mean():.3f}", end=", ")
    truth, prediction = predict(val)
    print(f"validation accuracy {(truth == prediction).mean():.3f}", flush=True)

print(classification_report(truth, prediction, target_names=CLASSES, digits=3))
