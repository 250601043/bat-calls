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
labels = torch.tensor(y.label.to_numpy())


def loader(rows, train):
    rows = torch.tensor(rows.to_numpy())
    return DataLoader(TensorDataset(x[rows], labels[rows]), batch_size=64, shuffle=train)


def run(images):
    images = images.to(device).float() / 40 + 1
    return model(images[:, None].repeat(1, 3, 1, 1))


def predict(data):
    model.eval()
    truth, prediction = [], []
    with torch.no_grad():
        for images, target in data:
            truth += target.tolist()
            prediction += run(images).argmax(1).tolist()
    return np.array(truth), np.array(prediction)


all_truth, all_prediction = [], []
for fold in range(5):
    train = loader((y.fold >= 0) & (y.fold != fold), True)
    val = loader(y.fold == fold, False)
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(512, len(CLASSES))
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        model.train()
        for images, target in train:
            loss = loss_function(run(images), target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        truth, prediction = predict(train)
        print(f"fold {fold + 1}, epoch {epoch + 1}: train accuracy {(truth == prediction).mean():.3f}", end=", ")
        truth, prediction = predict(val)
        print(f"validation accuracy {(truth == prediction).mean():.3f}", flush=True)
    all_truth += truth.tolist()
    all_prediction += prediction.tolist()

print(classification_report(all_truth, all_prediction, target_names=CLASSES, digits=3))
