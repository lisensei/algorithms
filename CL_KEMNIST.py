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
parser.add_argument("-learning_rate", default=1e-2)
parser.add_argument("-curriculum", type=int, default=1)
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
        self.additor = additor

    def __call__(self, label):
        label = label + self.additor
        return label


class KEMNIST(Dataset):
    def __init__(self, classes=[26, 10], batch_size=hp.batch_size, half_valsize=10000, mix_set_total=110000,
                 kmnist_len=50000):
        super(KEMNIST, self).__init__()
        self.batch_size = batch_size
        self.half_size = half_valsize
        self.mix_set_total = mix_set_total
        self.classes = classes
        self.total_classes = sum(classes)
        self.kmnist_len = kmnist_len

    def val_test(self):
        emnist_test = datarepo.EMNIST(root="./data", split="letters", train=False, download=True,
                                      transform=transforms.ToTensor(), target_transform=AddToLabel(-1))
        kmnist_test = datarepo.KMNIST(root="./data", train=False, download=True, transform=transforms.ToTensor(),
                                      target_transform=AddToLabel(self.classes[0]))
        mix_test_data = datautils.ConcatDataset([emnist_test, kmnist_test])
        mixed_test = DataLoader(dataset=mix_test_data,
                                shuffle=True, batch_size=self.batch_size)
        emnist_validation_indices = torch.arange(self.mix_set_total, self.mix_set_total + self.half_size,
                                                 dtype=torch.int32)
        kmnist_validation_indices = torch.arange(self.kmnist_len, self.kmnist_len + self.half_size,
                                                 dtype=torch.int32)
        validation_emnist = datautils.Subset(
            datarepo.EMNIST(root="./data", split="letters", train=True, download=True,
                            transform=transforms.ToTensor()),
            indices=emnist_validation_indices)
        validation_kmnist = datautils.Subset(
            datarepo.KMNIST(root="./data", train=True, download=True,
                            transform=transforms.ToTensor(), target_transform=AddToLabel(self.classes[0])),
            indices=kmnist_validation_indices)
        mixed_validation_data = datautils.ConcatDataset([validation_emnist, validation_kmnist])
        mixed_validation = DataLoader(
            dataset=mixed_validation_data, shuffle=True,
            batch_size=self.batch_size)
        self.test_size = len(mix_test_data)
        self.validation_size = len(mixed_validation_data)
        return mixed_test, mixed_validation

    def mix_dataset(self, kmnist_percent):
        num_kmnist_samples = self.kmnist_len * kmnist_percent / 100
        num_emnist_samples = self.mix_set_total - num_kmnist_samples
        emnist_indices = torch.arange(0, num_emnist_samples, dtype=torch.int32)
        kmnist_indices = torch.arange(0, num_kmnist_samples, dtype=torch.int32)
        emnist_samples = datautils.Subset(
            dataset=datarepo.EMNIST(root="./data", split="letters", train=True, download=True,
                                    transform=transforms.ToTensor(), target_transform=AddToLabel(-1)),
            indices=emnist_indices)
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

data_val, data_test = data.val_test()
increments = hp.mix_percent / hp.epochs

'''
Experiment repeats starts, the number of repeats is determined by the hyper parameter hp.repeats
'''
header = ["sequence", "repeat id", "batch size", "epoch", "curriculum", "current kmnist percent",
          "number of train correct", "training loss",
          "training accuracy", "number of test correct", "test accuracy", "validation accuracy"]

for repeat in range(hp.repeats):
    loss_function = nn.CrossEntropyLoss()
    net = CNN(data.total_classes)
    net.to(device=device)
    optimizer = torch.optim.SGD(net.parameters(), lr=hp.learning_rate)
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
        metrics["training loss"] = 0
        metrics["training accuracy"] = 0
        metrics["test accuracy"] = 0
        metrics["validation accuracy"] = 0
        metrics["current kmnist percent"] = 0
        metrics["repeat id"] = repeat
        metrics["number of train correct"] = 0
        metrics["number of test correct"] = 0
        metrics["epoch"] = epoch
        if hp.curriculum:
            metrics["curriculum"] = "true"
        else:
            metrics["curriculum"] = "false"

        '''
        Iterates overall three datasets.
        '''
        if epoch == hp.epochs - 1:
            datasets.append(data_val)
        for i, dataset in enumerate(datasets):
            samples = 0
            epoch_correct = 0
            epoch_loss = 0
            accuracy = 0
            if hp.curriculum and i == 0:
                dataset = data.mix_dataset(current_kmnist_percent)
                current_kmnist_percent += increments
            for j, (x, y) in enumerate(dataset):
                x = x.to(device)
                y = y.to(device)
                y_pred = net(x)
                loss = loss_function(y_pred, y.reshape(-1))
                if i == 0:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                epoch_loss += loss.detach()
                indices = torch.argmax(y_pred.detach(), dim=1).reshape(-1)
                num_correct = torch.sum(torch.tensor([1 if x else 0 for x in indices == y.detach()]))
                epoch_correct += num_correct
                samples += len(x.detach())
            metrics["current kmnist percent"] = current_kmnist_percent
            metrics["epoch"] = epoch
            if i == 0:
                accuracy = epoch_correct / samples
                metrics["number of train correct"] = epoch_correct.item()
                metrics["training accuracy"] = accuracy.item()
                metrics["training loss"] = epoch_loss.item()
            elif i == 1:
                accuracy = epoch_correct / samples
                metrics["number of test correct"] = epoch_correct.item()
                metrics["test accuracy"] = accuracy.item()
            else:
                accuracy = epoch_correct / data.validation_size
                metrics["validation accuracy"] = accuracy.item()
        epoch_df = epoch_df.append(metrics, ignore_index=True)
        print(
            f'repeat:{repeat},epoch:{epoch},train loss:{metrics["training loss"]},'
            f'training accuracy:{metrics["training accuracy"]},'
            f'test accuracy:{metrics["test accuracy"]},'
            f'validation accuracy:{metrics["validation accuracy"]}')
        if hp.curriculum:
            ending = "_with curriculum"
        else:
            ending = "_without curriculum"
        filename = "results/experiment_kemnist/metrics/sequence " + str(hp.sequence) + "/sequence " + str(
            hp.sequence) + "_repeat " + str(
            repeat) + "_mix percent " + str(
            hp.mix_percent) + ending + ".csv"
        epoch_df.to_csv(filename, index=False)
