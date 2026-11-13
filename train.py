import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ["Biting", "Feeding", "Fighting", "Mating protest", "Sleeping"]
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0)
torch.manual_seed(0)

x = torch.from_numpy(np.load("data/x.npy"))
y = pd.read_csv("data/y.csv")
labels = torch.tensor(y.label.to_numpy())


def loader(rows, train):
    rows = torch.tensor(rows.to_numpy())
    return DataLoader(TensorDataset(x[rows], labels[rows]), batch_size=64, shuffle=train)


def blank_strips(images):
    for image in images:
        f, t = random.randrange(112), random.randrange(240)
        image[f:f + random.randrange(17)] = -80
        image[:, t:t + random.randrange(17)] = -80
    return images


def run(images):
    images = images.to(device).float() / 40 + 1
    return model(images[:, None].repeat(1, 3, 1, 1))


def predict(data):
    model.eval()
    truth, guesses = [], []
    with torch.no_grad():
        for images, target in data:
            truth += target.tolist()
            guesses += run(images).topk(2).indices.tolist()
    return np.array(truth), np.array(guesses)


def accuracy(truth, guesses, n):
    return (truth[:, None] == guesses[:, :n]).any(1).mean()


all_truth, all_guesses = [], []
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
            loss = loss_function(run(blank_strips(images)), target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        truth, guesses = predict(train)
        print(f"fold {fold + 1}, epoch {epoch + 1}: train accuracy {accuracy(truth, guesses, 1):.3f}", end=", ")
        truth, guesses = predict(val)
        print(f"validation accuracy {accuracy(truth, guesses, 1):.3f}, within two guesses {accuracy(truth, guesses, 2):.3f}", flush=True)
    all_truth += truth.tolist()
    all_guesses += guesses.tolist()

truth, guesses = np.array(all_truth), np.array(all_guesses)
counts = np.sort(np.bincount(truth))[::-1]
print(classification_report(truth, guesses[:, 0], target_names=CLASSES, digits=3))
print(f"one guess: model {accuracy(truth, guesses, 1):.3f}, blind guess {counts[0] / len(truth):.3f}")
print(f"two guesses: model {accuracy(truth, guesses, 2):.3f}, blind guess {counts[:2].sum() / len(truth):.3f}")
for i, name in enumerate(CLASSES):
    print(f"{name} within two guesses: {accuracy(truth[truth == i], guesses[truth == i], 2):.3f}")
