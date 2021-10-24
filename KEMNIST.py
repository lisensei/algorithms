import torch
import torch.nn as nn
import argparse
import torchvision.datasets as datarepo
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
import numpy as np

parser = argparse.ArgumentParser(description="hyper parameters")
parser.add_argument("-mix_percent", default=1)
parser.add_argument("-learning_rate", default=3e-1)
parser.add_argument("-curriculum", type=int, default=0)
parser.add_argument("-repeats", type=int, default=10)
parser.add_argument("-epochs", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-max_percent", type=float, default=0)
parser.add_argument("-sequence", type=int, default=1)
hp = parser.parse_args()


class KEMNIST(Dataset):
    def __init__(self, val_percent=1 / 5, classes=torch.tensor([26, 10])):
        super(KEMNIST, self).__init__()
        self.mix_set_total = 110000
        self.classes = classes
        self.total_classes = torch.sum(self.classes)
        self.emnist_train = datarepo.EMNIST(root="./data", split="letters", train=True, download=True,
                                            transform=ToTensor())
        self.kmnist_train = datarepo.KMNIST(root="./data", train=True, download=True)
        self.emnist_test = datarepo.EMNIST(root="./data", split="letters", train=False, download=True,
                                           transform=ToTensor())
        self.kmnist_test = datarepo.KMNIST(root="./data", train=False, download=True, transform=ToTensor())
        self.emnist_len = len(self.emnist_train)
        self.kmnist_len = len(self.kmnist_train)
        self.mixed_train = self.mix_dataset(0)
        self.mixed_test = TensorDataset(torch.cat((self.emnist_test.data, self.kmnist_test.data)),
                                        torch.cat(
                                            (self.emnist_test.targets, self.kmnist_test.targets + self.classes[0])))
        self.mixed_validation = TensorDataset(
            torch.cat((self.emnist_train.data[114800:], self.kmnist_train.data[50000:])),
            torch.cat((self.emnist_train.targets[114800:], self.kmnist_train.targets[50000:] + self.classes[0])))
        self.test_size = len(self.mixed_test)
        self.validation_size = len(self.mixed_validation)

    def get_dataset(self, dataset_type):
        if dataset_type not in ["train", "validation", "test"]:
            print("Wrong dataset_type")
            return
        if dataset_type == "train":
            return self.mixed_train
        elif dataset_type == "validation":
            return self.mixed_validation
        else:
            return self.mixed_test

    def mix_dataset(self, kmnist_percent):
        num_kmnist_samples = self.kmnist_len * kmnist_percent
        num_emnist_samples = self.mix_set_total - num_kmnist_samples
        mixed_dataset_train = TensorDataset(torch.cat((self.emnist_train.data[:num_emnist_samples],
                                                       self.kmnist_train.data[:num_kmnist_samples])),
                                            torch.cat((self.emnist_train.targets[:num_emnist_samples],
                                                       self.kmnist_train.targets[:num_kmnist_samples])))
        mixed_dataset_train = DataLoader(mixed_dataset_train, shuffle=True, batch_size=hp.batch_size)
        return mixed_dataset_train


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.covs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=13, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=13, out_channels=21, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=21, out_channels=34, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=34),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=34, out_channels=55, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=55, out_channels=89, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=89, out_channels=144, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=144),
            nn.MaxPool2d(kernel_size=2),
        )
        self.first_liner_layer = nn.Linear(in_features=1296, out_features=134)
        self.fc = nn.Linear(in_features=134, out_features=num_classes)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        output = self.covs.forward(x)
        output = output.reshape(len(x), -1)
        output = self.first_liner_layer(output)
        output = self.fc(output)
        return output


torch.manual_seed(np.e)
random.seed(np.e)
ok = random.randint(5, 10000)

data = KEMNIST()
if hp.curriculum:
    data_train = data.mix_dataset(0)
else:
    data_train = data.mix_dataset(hp.mix_percent)

data_val = DataLoader(data.get_dataset("validation"), shuffle=True, batch_size=hp.batch_size)
data_test = DataLoader(data.get_dataset("test"), shuffle=True, batch_size=hp.batch_size)
loss_function = nn.CrossEntropyLoss()
net = CNN(data.total_classes)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
increments = hp.mix_percent / hp.epochs

for re in range(hp.repeats):
    current_mix_percent = 0
    if hp.curriculum:
        data_train = data.mix_dataset(0)
    datasets = [data_train, data_val, data_test]

    for dataset in datasets:
        if dataset == data_train:
            epoch = hp.epochs
        else:
            epoch = 1
        for e in range(epoch):
            epoch_correct = 0
            epoch_loss = 0
            accuracy = 0
            if hp.curriculum and dataset == data_train:
                dataset = data.mix_dataset(current_mix_percent)
                current_mix_percent += increments
            for i, (x, y) in enumerate(dataset):
                y_pred = net(x.float())
                indices = torch.argmax(y_pred, dim=1).reshape(-1)
                num_correct = torch.sum(torch.tensor([1 if x else 0 for x in indices == y]))
                epoch_correct += num_correct
                loss = loss_function(y_pred, y.reshape(-1))
                epoch_loss += loss
                if dataset == data_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            if dataset == data_train:
                accuracy = epoch_correct / data.mix_set_total
                print(f"epoch:{e},train loss:{epoch_loss},training accuracy:{accuracy}")
            elif dataset == data_val:
                accuracy = epoch_correct / data.validation_size
                print(f"epoch:{e},validation loss:{epoch_loss},validation accuracy:{accuracy}")
            else:
                accuracy = epoch_correct / data.test_size
                print(f"epoch:{e},test loss:{epoch_loss},test accuracy:{accuracy}")
