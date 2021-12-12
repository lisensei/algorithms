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
import os

parser = argparse.ArgumentParser(description="hyper parameters")
parser.add_argument("-km_classes", type=int, default=10)
parser.add_argument("-learning_rate", default=1e-2)
parser.add_argument("-curriculum", type=int, default=0)
parser.add_argument("-repeats", type=int, default=10)
parser.add_argument("-epochs", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-sequence", type=int, default=1)
parser.add_argument("-samples_per_class", type=int, default=500)
hp = parser.parse_args([])

'''
Custom transform to adjust the numbering of classes in the mixed datasets. 
'''


class AddToLabel:
    def __init__(self, additor):
        self.additor = additor

    def __call__(self, label):
        label = label + self.additor
        return torch.tensor(label)


class KEMNIST(Dataset):
    def __init__(self, classes=[26, 10], batch_size=hp.batch_size, val_size=10000, test_size=10000,
                 mix_set_total=100000,
                 kmnist_len=50000):
        super(KEMNIST, self).__init__()
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.mix_set_total = mix_set_total
        self.classes = classes
        self.total_classes = sum(classes)
        self.kmnist_len = kmnist_len

    '''
    Generates the validation set and test set.
    '''

    def val_test(self, kmnist_classes, samples_per_class=hp.samples_per_class):
        emnist_indices = torch.arange(0, self.mix_set_total, dtype=torch.int32)
        emnist_valsamples = datautils.Subset(
            dataset=datarepo.EMNIST(root="./data", train=True, split="letters", download=True,
                                    transform=transforms.ToTensor(),
                                    target_transform=AddToLabel(-1)),
            indices=emnist_indices)
        emnist_testsamples = datarepo.EMNIST(root="./data", train=False, split="letters", download=True,
                                             transform=transforms.ToTensor(),
                                             target_transform=AddToLabel(-1))
        kmnist_valsamples = datarepo.KMNIST(root="./data", train=True, download=True, transform=transforms.ToTensor(),
                                            target_transform=AddToLabel(self.classes[0]))
        kmnist_testsamples = datarepo.KMNIST(root="./data", train=False, download=True,
                                             transform=AddToLabel(self.classes[0]))
        kmnist_valdata = kmnist_valsamples.data[self.kmnist_len:]
        kmnist_vallabel = kmnist_valsamples.targets[self.kmnist_len:]
        kmnist_validation_set = [(x, y) for x, y in zip(kmnist_valdata, kmnist_vallabel)]
        kmnist_validation_set.sort(key=lambda x: x[1])
        kmnist_validation_datalist = []
        kmnist_validation_labellist = []
        for i in range(len(kmnist_valdata)):
            kmnist_validation_datalist.append(kmnist_validation_set[i][0].unsqueeze(dim=0))
            kmnist_validation_labellist.append(kmnist_validation_set[i][1])

        '''
        Count the samples of different classes, get corresponding offset and generate final validation set
        '''
        validation_classes = list(torch.tensor(kmnist_validation_labellist).tolist())
        val_offset = torch.tensor([validation_classes.count(i) for i in set(validation_classes)]).cumsum(dim=0).tolist()
        val_offset.insert(0, 0)

        kmnist_validation_data = []
        kmnist_validation_label = []
        for j in range(kmnist_classes):
            for k in range(samples_per_class):
                kmnist_validation_data.append(kmnist_validation_datalist[val_offset[j] + k])
                kmnist_validation_label.append(kmnist_validation_labellist[val_offset[j] + k])

        kmnist_validation_data = torch.cat(kmnist_validation_data).unsqueeze(dim=1)
        kmnist_validation_label = torch.tensor(kmnist_validation_label) + self.classes[0]
        kmnist_validation_set = datautils.TensorDataset(kmnist_validation_data, kmnist_validation_label)

        '''mixed test set'''
        kmnist_testdata = kmnist_testsamples.data
        kmnist_testlabel = kmnist_testsamples.targets
        kmnist_test_set = [(x, y) for x, y in zip(kmnist_testdata, kmnist_testlabel)]
        kmnist_test_set.sort(key=lambda x: x[1])
        kmnist_test_datalist = []
        kmnist_test_labellist = []
        for i in range(len(kmnist_testdata)):
            kmnist_test_datalist.append(kmnist_test_set[i][0].unsqueeze(dim=0))
            kmnist_test_labellist.append(kmnist_test_set[i][1])

        '''
        Count the samples of different classes, get corresponding offset and generate final test set
        '''
        test_classes = list(torch.tensor(kmnist_test_labellist).tolist())
        test_offset = torch.tensor([test_classes.count(i) for i in set(test_classes)]).cumsum(dim=0).tolist()
        test_offset.insert(0, 0)

        kmnist_test_data = []
        kmnist_test_label = []
        for j in range(kmnist_classes):
            for k in range(samples_per_class):
                kmnist_test_data.append(kmnist_test_datalist[test_offset[j] + k])
                kmnist_test_label.append(kmnist_test_labellist[test_offset[j] + k])

        kmnist_test_data = torch.cat(kmnist_test_data).unsqueeze(dim=1)
        kmnist_test_label = torch.tensor(kmnist_test_label) + self.classes[0]
        kmnist_test_set = datautils.TensorDataset(kmnist_test_data, kmnist_test_label)
        mixed_validation_set = DataLoader(dataset=datautils.ConcatDataset([emnist_valsamples, kmnist_validation_set]),
                                          shuffle=True,
                                          batch_size=self.batch_size)
        mixed_test_set = DataLoader(dataset=datautils.ConcatDataset([emnist_testsamples, kmnist_test_set]),
                                    shuffle=True,
                                    batch_size=self.batch_size)
        return mixed_validation_set, mixed_test_set

    def mix_dataset(self, kmclasses):
        num_kmnist_samples = int(self.kmnist_len * kmclasses / 10)
        num_emnist_samples = self.mix_set_total - num_kmnist_samples
        emnist_indices = torch.arange(0, num_emnist_samples, dtype=torch.int32)
        emnist_samples = datautils.Subset(
            dataset=datarepo.EMNIST(root="./data", split="letters", train=True, download=True,
                                    transform=transforms.ToTensor(), target_transform=AddToLabel(-1)),
            indices=emnist_indices)
        kmnist_samples = datarepo.KMNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True,
                                         target_transform=AddToLabel(self.classes[0]))
        kmnist_samples = [(x, y) for x, y in zip(kmnist_samples.data.to(torch.float32)[:self.kmnist_len],
                                                 kmnist_samples.targets[:self.kmnist_len])]
        kmnist_samples.sort(key=lambda x: x[1])
        km_data = []
        km_label = []
        for i in range(num_kmnist_samples):
            km_data.append(kmnist_samples[i][0].unsqueeze(dim=0))
            km_label.append(kmnist_samples[i][1])
        km_data = torch.cat(km_data, dim=0).unsqueeze(dim=1)
        km_label = torch.tensor(km_label)
        kmnist_samples = datautils.TensorDataset(km_data, km_label)
        mixedset = datautils.ConcatDataset([emnist_samples, kmnist_samples])
        mixed_dataset_train = DataLoader(dataset=mixedset,
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
    data_train = data.mix_dataset(hp.km_classes)

data_val, data_test = data.val_test(hp.km_classes)
step_size = hp.km_classes / hp.epochs

'''
Experiment repeats starts, the number of repeats is determined by the hyper parameter hp.repeats
'''
header = ["sequence", "repeat id", "batch size", "epoch", "curriculum", "current kmnist classes",
          "number of train correct", "training loss",
          "training accuracy", "number of test correct", "test accuracy", "validation accuracy"]

for repeat in range(hp.repeats):
    loss_function = nn.CrossEntropyLoss()
    net = CNN(data.total_classes)
    net.to(device=device)
    optimizer = torch.optim.SGD(net.parameters(), lr=hp.learning_rate)
    current_kmnist_classes = 0
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
        metrics["current kmnist classes"] = 0
        metrics["repeat id"] = repeat
        metrics["number of train correct"] = 0
        metrics["number of test correct"] = 0
        metrics["epoch"] = epoch
        if hp.curriculum:
            metrics["curriculum"] = "true"
        else:
            metrics["curriculum"] = "false"

        '''
        Iterates over all three datasets.
        '''
        if epoch == hp.epochs - 1:
            datasets.append(data_val)
        for i, dataset in enumerate(datasets):
            samples = 0
            epoch_correct = 0
            epoch_loss = 0
            accuracy = 0
            if hp.curriculum and i == 0 and epoch != 0 and epoch % step_size == 0:
                dataset = data.mix_dataset(current_kmnist_classes)
                current_kmnist_classes += 1
            if not hp.curriculum:
                current_kmnist_classes = hp.km_classes
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
            metrics["current kmnist classes"] = current_kmnist_classes
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
                accuracy = epoch_correct / samples
                metrics["validation accuracy"] = accuracy.item()
        epoch_df = epoch_df.append(metrics, ignore_index=True)
        print(
            f'repeat:{repeat},epoch:{epoch},current kmnist classes:{metrics["current kmnist classes"]},train loss:{metrics["training loss"]},'
            f'training accuracy:{metrics["training accuracy"]},'
            f'test accuracy:{metrics["test accuracy"]},'
            f'validation accuracy:{metrics["validation accuracy"]}')
        if hp.curriculum:
            ending = "_with curriculum"
        else:
            ending = "_without curriculum"
        filedir = filename = "results/experiment_kemnist/metrics/sequence " + str(hp.sequence)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        filename = filedir + "/sequence " + str(
            hp.sequence) + "_repeat " + str(
            repeat) + "_max kmclass" + str(
            hp.km_classes) + ending + ".csv"

        epoch_df.to_csv(filename, index=False)
