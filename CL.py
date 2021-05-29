import argparse
import csv

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.utils
import torch.utils
import torch.utils
import torch.utils.data as Data
import torchvision.models
import torchvision.transforms as transform
import numpy as np
# Hyper parameters
parser = argparse.ArgumentParser(description="Hyper parameters")
parser.add_argument("-run_name", default="0")
parser.add_argument("-sequence_name", default="cl1")
parser.add_argument("-learning_rate", default=3e-3)
parser.add_argument("-epoches", default=20)
parser.add_argument("-batch_size", default=16)
parser.add_argument("-noise_percent", default=0)
parser.add_argument("-mean", default=0)
parser.add_argument("-std", default=1)
parser.add_argument("-classes", default=10)

hp = parser.parse_args()
filename = "sequence_" + str(hp.sequence_name) + "_" + "run_name " + str(
    hp.run_name) + "_" + " metrics_with_curriculum_cl.csv"

sequence_result = "aggregated sequence result " + str(hp.sequence_name)
with open(filename, 'a', newline='') as csvfile:
    fwrite = csv.writer(csvfile, delimiter=',')
    fwrite.writerow(["Sequence", "Run Name", "Learning Rate", "Batch Size", "Noise Percent",
                     "Epoch", "Train Loss", "Train Accuracy", "Train F1",
                     "Test Loss", "Test Accuracy", "Test F1",
                     "Train Confusion Matrix", "Test Confusition Matrix"])

with open(sequence_result, 'a', newline='') as csvfile:
    fwrite = csv.writer(csvfile, delimiter=',')
    fwrite.writerow(["Sequence", "Run Name", "Learning Rate", "Batch Size", "Noise Percent",
                     "At Epoch", "Best Test Loss", "Best Test Accuracy", "Test F1"])


def addPeperNoise(source, percent):
    source = source.squeeze(dim=1)
    for image in source:
        for j in range(int(255 * percent / 100)):
            x = torch.randint(0, 28, (1,))
            y = torch.randint(0, 28, (1,))
            if (x + y) % 2 == 0:
                image[x, y] = 0
            else:
                image[x, y] = 255
    return source.unsqueeze(dim=1)


class TransformNoise:
    def __init__(self):
        self.percent_noise = 0.0

    def __call__(self, x):
        return addPeperNoise(x, self.percent_noise)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def Fonescore(matrix):
    matrix_size = len(matrix)
    parameters = torch.zeros((matrix_size, 3))
    for i in range(matrix_size):
        parameters[i][0] = matrix[i, i]
        for j in range(matrix_size):
            if j != i:
                parameters[i][1] += matrix[j, i]
                parameters[i][2] += matrix[i, j]

    precision = 0
    recall = 0
    for i in range(matrix_size):
        precision += parameters[i][0] / (parameters[i][0] + parameters[i][1]) / matrix_size
        recall += parameters[i][0] / (parameters[i][0] + parameters[i][2]) / matrix_size

    f1score = 2 * precision * recall / (precision + recall)
    return f1score


# Data processing
curriculum_noise = TransformNoise()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_train = torchvision.datasets.MNIST(root='/data', train=True, transform=transform.Compose(
    [
        transform.ToTensor(),
        curriculum_noise,
        torchvision.transforms.Normalize(mean=0.0, std=1.0)
    ]
), download=True)
data_test = torchvision.datasets.MNIST(root='/data', train=False, transform=transform.Compose(
    [
        transform.ToTensor(),
        curriculum_noise,
        torchvision.transforms.Normalize(mean=hp.mean, std=hp.std)
    ]
), download=True)
train_set = Data.DataLoader(
    dataset=Data.TensorDataset(data_train.data.to(torch.float32).unsqueeze(dim=1),
                               data_train.targets), batch_size=hp.batch_size, shuffle=True)
test_set = Data.DataLoader(
    dataset=Data.TensorDataset(data_test.data.to(torch.float32).unsqueeze(dim=1),
                               data_test.targets), batch_size=hp.batch_size, shuffle=False)


class ResBlock(nn.Module):
    def __init__(self, in_channels, middle_out, out_channels):
        super(ResBlock, self).__init__()
        self.upper_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                   out_channels=middle_out,
                                                   kernel_size=3,
                                                   padding=1),
                                         nn.ReLU())
        self.lower_layer = nn.Sequential(nn.Conv2d(in_channels=middle_out,
                                                   out_channels=out_channels,
                                                   kernel_size=3,
                                                   padding=1),
                                         nn.ReLU())

    def forward(self, x):
        z = self.upper_layer.forward(x)
        z_out = self.lower_layer.forward(z)
        return z_out + x


class ResNet(nn.Module):
    def __init__(self, number_of_class):
        self.batch_size = hp.batch_size
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(
            ResBlock(1, 2, 4),
            ResBlock(4, 4, 4),
        )
        self.cov = nn.Conv2d(in_channels=4,
                             out_channels=8,
                             kernel_size=3,
                             padding=1)
        self.relu = nn.ReLU()
        self.linear_layer = nn.Linear(in_features=28 * 28 * 8, out_features=number_of_class)

    def forward(self, x):
        out = self.layers.forward(x)
        out = self.cov(out)
        out = self.relu(out)
        out = out.reshape(self.batch_size, 28 * 28 * 8)
        out = self.linear_layer.forward(out)
        return out


net = ResNet(hp.classes)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=hp.learning_rate)
is_curriculum_enabled = False
if not is_curriculum_enabled:
    curriculum_noise.percent_noise = 50

all_metrics = list()
test_accuracies=torch.zeros((hp.epoches,1))
for iteration in range(hp.epoches):
    metrics = []
    net.batch_size = hp.batch_size

    # Train F1 variables and confusion matrix declaration
    confusion_matrix_train = torch.zeros((10, 10))
    for i, dataloader in enumerate([train_set, test_set]):
        train_loss = 0
        train_correct = 0
        confusion_matrix_train = torch.zeros((10, 10))
        for j, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            ypredict = net.forward(x).to(torch.float32)
            indices = torch.argmax(ypredict, dim=1)
            for row, column in zip(y, indices):
                confusion_matrix_train[row, column] += 1
            train_correct += ((indices - y) == 0).sum()
            batch_loss = loss.forward(ypredict, y)
            train_loss += batch_loss
            if dataloader == train_set:
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if j % 1000 == 0:
                print("epoch:", iteration, "batch:", j, "Loss:", batch_loss.item())
        train_accuracy = train_correct / len(dataloader.dataset)
        f1score_train = Fonescore(confusion_matrix_train)
        metrics.append(train_loss)
        metrics.append(train_accuracy)
        metrics.append(f1score_train)
        metrics.append(confusion_matrix_train)

    all_metrics.append(metrics)
    test_accuracies[iteration] = metrics[5]
    print(
        f'Train loss:{metrics[0]:.4f}, Train accuracy:{metrics[1]:.4f},'
        f'train F1 score:{metrics[2]:.4f},Test loss:{metrics[4]:.4f},'
        f'Test accuracy:{metrics[5]:.4f},Test f1 score:{metrics[6]:.4f}')

    with open(filename, 'a', newline='') as csvfile:
        fwrite = csv.writer(csvfile, delimiter=',')
        fwrite.writerow(
            [hp.sequence_name, hp.run_name, hp.learning_rate, hp.batch_size, hp.noise_percent, iteration, metrics[0],
             metrics[1],
             metrics[2], metrics[3],
             metrics[4], metrics[5], metrics[6], metrics[7]])

at_epoch=torch.argmax(test_accuracies)
with open(sequence_result, 'a', newline='') as csvfile:
    fwrite = csv.writer(csvfile, delimiter=',')
    fwrite.writerow([hp.sequence_name, hp.run_name, hp.learning_rate, hp.batch_size, hp.noise_percent,
                             at_epoch, all_metrics[at_epoch][4], all_metrics[at_epoch][5], all_metrics[at_epoch][6]])
