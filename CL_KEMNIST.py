import torch
import torch.nn as nn
import argparse
import torchvision.datasets as datarepo
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as datautils
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="hyper parameters")
parser.add_argument("-mix_percent", type=int, default=1)
parser.add_argument("-learning_rate", default=3e-1)
parser.add_argument("-curriculum", type=int, default=0)
parser.add_argument("-repeats", type=int, default=1)
parser.add_argument("-epochs", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-sequence", type=int, default=1)
hp = parser.parse_args()

'''
Custom transform to adjust the numbering of classes in the mixed datasets. 
'''


class AddToLabel:
    def __init__(self, additor):
        self.addditor = additor

    def __call__(self, label):
        label = label + self.addditor
        return label


class KEMNIST(Dataset):
    def __init__(self, classes=[26, 10], batch_size=16, half_valsize=10000, mix_set_total=110000, kmnist_len=50000):
        super(KEMNIST, self).__init__()
        self.batch_size = batch_size
        self.half_size = half_valsize
        self.mix_set_total = mix_set_total
        self.classes = classes
        self.total_classes = sum(classes)
        self.kmnist_len = kmnist_len
        self.emnist_test = datarepo.EMNIST(root="./data", split="letters", train=False, download=True,
                                           transform=transforms.ToTensor())
        self.kmnist_test = datarepo.KMNIST(root="./data", train=False, download=True, transform=transforms.ToTensor(),
                                           target_transform=AddToLabel(self.classes[0]))
        self.mix_test_data = datautils.ConcatDataset([self.emnist_test, self.kmnist_test])
        self.mixed_test = DataLoader(dataset=self.mix_test_data,
                                     shuffle=True, batch_size=self.batch_size)
        self.emnist_validation_indices = torch.arange(self.mix_set_total, self.mix_set_total + self.half_size,
                                                      dtype=torch.int32)
        self.kmnist_validation_indices = torch.arange(self.kmnist_len, self.kmnist_len + self.half_size,
                                                      dtype=torch.int32)
        self.validation_emnist = datautils.Subset(
            datarepo.EMNIST(root="./data", split="letters", train=True, download=True,
                            transform=transforms.ToTensor()),
            indices=self.emnist_validation_indices)
        self.validation_kmnist = datautils.Subset(
            datarepo.KMNIST(root="./data", train=True, download=True,
                            transform=transforms.ToTensor(), target_transform=AddToLabel(self.classes[0])),
            indices=self.kmnist_validation_indices)
        self.mixed_validation_data = datautils.ConcatDataset([self.validation_emnist, self.validation_kmnist])
        self.mixed_validation = DataLoader(
            dataset=self.mixed_validation_data, shuffle=True,
            batch_size=self.batch_size)
        self.test_size = len(self.mix_test_data)
        self.validation_size = len(self.mixed_validation_data)

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
        num_kmnist_samples = self.kmnist_len * kmnist_percent / 100
        num_emnist_samples = self.mix_set_total - num_kmnist_samples
        emnist_indices = torch.arange(0, num_emnist_samples, dtype=torch.int32)
        kmnist_indices = torch.arange(0, num_kmnist_samples, dtype=torch.int32)
        emnist_samples = datautils.Subset(
            dataset=datarepo.EMNIST(root="./data", split="letters", train=True, download=True,
                                    transform=transforms.ToTensor()), indices=emnist_indices)
        kmnist_samples = datautils.Subset(
            dataset=datarepo.KMNIST(root="./data", train=True, transform=transforms.ToTensor(),
                                    target_transform=AddToLabel(self.classes[0])), indices=kmnist_indices)
        mixed_dataset_train = DataLoader(dataset=datautils.ConcatDataset([emnist_samples, kmnist_samples]),
                                         shuffle=True, batch_size=self.batch_size)
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
        output = self.covs.forward(x)
        output = output.reshape(len(x), -1)
        output = self.first_liner_layer(output)
        output = self.fc(output)
        return output


torch.manual_seed(np.e)
random.seed(np.e)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
data = KEMNIST()
if hp.curriculum:
    data_train = data.mix_dataset(0)
else:
    data_train = data.mix_dataset(hp.mix_percent)

data_val = data.get_dataset("validation")
data_test = data.get_dataset("test")
increments = hp.mix_percent / hp.epochs

'''
Experiment repeats starts, the number of repeats is determined by the hyper parameter hp.repeats
'''
header = ["sequence", "repeat id", "batch size", "epoch", "curriculum", "current kmnist percent",
          "number of correct", "training loss",
          "training accuracy", "test accuracy", "validation accuracy"]

for repeat in range(hp.repeats):
    loss_function = nn.CrossEntropyLoss()
    net = CNN(data.total_classes)
    net.to(device=device)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
    current_kmnist_percent = 0
    if hp.curriculum:
        data_train = data.mix_dataset(0)
    datasets = [data_train, data_test]

    '''
        Epochs starts, number of epochs for the three datasets are different.
        The number of epochs for the training set is given by the hyper parameter hp.epochs.
        The number of epochs for the validation set and test set are on.
        '''
    epoch_df = pd.DataFrame(columns=header)
    for epoch in range(hp.epochs):
        metrics = dict()
        metrics["sequence"] = hp.sequence
        metrics["batch size"] = hp.batch_size
        metrics["training accuracy"] = 0
        metrics["test accuracy"] = 0
        metrics["validation accuracy"] = 0
        metrics["current kmnist percent"] = 0
        metrics["repeat id"] = repeat
        metrics["number of correct"] = 0
        metrics["epoch"] = epoch
        if hp.curriculum:
            metrics["curriculum"] = "true"
        else:
            metrics["curriculum"] = "false"

        '''
        Iterates overall three datasets.
        '''
        for dataset in datasets:
            if epoch == hp.epochs - 1:
                datasets.append(data_val)
            epoch_correct = 0
            epoch_loss = 0
            accuracy = 0
            if hp.curriculum and dataset == data_train:
                dataset = data.mix_dataset(current_kmnist_percent)
                current_kmnist_percent += increments
            for i, (x, y) in enumerate(dataset):
                x = x.to(device).detach()
                y = y.to(device).detach()
                y_pred = net(x.float())
                loss = loss_function(y_pred, y.reshape(-1))
                epoch_loss += loss
                if dataset == data_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                indices = torch.argmax(y_pred, dim=1).reshape(-1)
                num_correct = torch.sum(torch.tensor([1 if x else 0 for x in indices == y])).to(device)
                epoch_correct += num_correct
            metrics["number of correct"] = epoch_correct.item()
            metrics["current kmnist percent"] = current_kmnist_percent
            metrics["epoch"] = epoch
            metrics["training loss"] = epoch_loss.item()
            if dataset == data_train:
                accuracy = epoch_correct / data.mix_set_total
                metrics["training accuracy"] = accuracy.item()
                print(f"epoch:{epoch},train loss:{epoch_loss},training accuracy:{accuracy}")
            elif dataset == data_test:
                accuracy = epoch_correct / data.test_size
                metrics["test accuracy"] = accuracy.item()
                print(f"epoch:{epoch},test loss:{epoch_loss},test accuracy:{accuracy}")
            else:
                accuracy = epoch_correct / data.validation_size
                metrics["validation accuracy"] = accuracy.item()
                print(f"epoch:{epoch},test loss:{epoch_loss},test accuracy:{accuracy}")
            epoch_df = epoch_df.append(metrics, ignore_index=True)
        if hp.curriculum:
            ending = "_with curriculum"
        else:
            ending = "_without curriculum"
        filename = "results/experiment_kemnist/metrics/sequence " + str(hp.sequence) + "/sequence " + str(
            hp.sequence) + "_repeat " + str(
            repeat) + "_mix percent " + str(
            hp.mix_percent) + ending + ".csv"
        epoch_df.to_csv(filename, index=False)
