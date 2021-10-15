import argparse
import csv
import os
import random

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.utils
import torch.utils
import torch.utils
import torch.utils.data as Data
import torchvision.models
import torchvision.transforms as transform
from torch.utils.tensorboard import SummaryWriter

'''Argparse hyper parameters'''

parser = argparse.ArgumentParser(description="Hyper parameters")
parser.add_argument("-run_name", default=0)
parser.add_argument("-sequence_name", default=1)
parser.add_argument("-learning_rate", default=3e-3)
parser.add_argument("-epoches", default=20)
parser.add_argument("-batch_size", default=16)
parser.add_argument("-noise_percent", default=1, type=int)
parser.add_argument("-mean", default=0)
parser.add_argument("-std", default=1)
parser.add_argument("-classes", default=10)
parser.add_argument("-curriculum", default=False)
parser.add_argument("-samples", default=1)
hp = parser.parse_args()

'''This class is used to process dataset,such as adding noise'''


class DataVisualizer:
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def addToTensorboard(self, figure, step):
        self.writer.add_figure(tag="confusion matrix", figure=figure, global_step=step)

    def plotMatrix(self, matrices):
        figure = plt.figure(figsize=[10, 5])

        for i in range(2):
            matrix = matrices[i]
            plt.subplot(1, 2, i + 1)
            if i == 0:
                plt.title("Training Set Confusion Matrix")
            else:
                plt.title("Test Set Confusion Matrix")
            plt.imshow(matrix, interpolation="nearest", cmap=plt.get_cmap("Blues"))
            plt.xticks(list(np.linspace(0, 9, 10, dtype=int)), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            plt.yticks(list(np.linspace(0, 9, 10)), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            for row in range(len(matrix[i])):
                for column in range(len(matrix)):
                    plt.annotate(str(int(matrix[row, column])), xy=(column, row),
                                 horizontalalignment="center",
                                 verticalalignment="center")
            plt.xlabel("Predicted")
            plt.ylabel("True")
        return figure


class DataProcessor:
    def __init__(self, trainset_size=5 / 6, valset_size=1 / 6, batch_size=16):
        self.batch_size = batch_size
        self.data_train = torchvision.datasets.MNIST(root='./data', train=True, transform=transform.Compose(
            [
                transform.ToTensor()
            ]
        ), download=True)
        self.data_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform.Compose(
            [
                transform.ToTensor(),
            ]
        ), download=True)
        max_samples = len(self.data_train.data)
        self.train_samples = int(max_samples * trainset_size)
        self.val_samples = int(max_samples * valset_size)
        start_index = max_samples - self.val_samples
        self.train_data = torch.clone(self.data_train.data[0:self.train_samples])
        self.train_target = torch.clone(self.data_train.targets[0:self.train_samples])
        self.validation_data = torch.clone(self.data_train.data[start_index:])
        self.validation_target = torch.clone(self.data_train.targets[start_index:])
        self.train_set = Data.DataLoader(
            dataset=Data.TensorDataset(self.train_data.to(torch.float32).unsqueeze(dim=1),
                                       self.train_target), batch_size=self.batch_size, shuffle=True)
        self.test_set = Data.DataLoader(
            dataset=Data.TensorDataset(self.data_test.data.to(torch.float32).unsqueeze(dim=1),
                                       self.data_test.targets), batch_size=self.batch_size, shuffle=False)
        self.validation_set = Data.DataLoader(
            dataset=Data.TensorDataset(self.validation_data.to(torch.float32).unsqueeze(dim=1),
                                       self.validation_target), batch_size=self.batch_size, shuffle=True)

    def repack(self):
        self.train_set = Data.DataLoader(
            dataset=Data.TensorDataset(self.train_data.to(torch.float32).unsqueeze(dim=1),
                                       self.train_target), batch_size=self.batch_size, shuffle=True)
        self.test_set = Data.DataLoader(
            dataset=Data.TensorDataset(self.data_test.data.to(torch.float32).unsqueeze(dim=1),
                                       self.data_test.targets), batch_size=self.batch_size, shuffle=False)

    def getDataset(self, ):
        return self.train_set, self.test_set

    def addPepperNoise(self, percent):
        self.resetDataset()
        self.train_data.squeeze(dim=1)
        if percent >= 100:
            percent = 100

        for i, image in enumerate(self.train_data):
            polluted_pixels = random.sample(range(0, 28 * 28), int(percent / 100 * 28 * 28))
            for i in range(len(polluted_pixels)):
                row = int(polluted_pixels[i] / 28)
                col = polluted_pixels[i] % 28
                if (row + col) % 2 == 0:
                    image[row][col] = 0
                else:
                    image[row][col] = 255.0
        self.repack()

    def resetDataset(self):
        self.train_data = torch.clone(self.data_train.data[0:self.train_samples])
        self.train_target = torch.clone(self.data_train.targets[0:self.train_samples])


'''This class is used to make statistical calculations such as calculate F1 score'''


class StatisticalAnalyzer:
    def fonescore(self, matrix):
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
    def __init__(self, number_of_class, batch_size=16):
        self.batch_size = batch_size
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
        out = out.reshape(len(out), 28 * 28 * 8)
        out = self.linear_layer.forward(out)
        return out


torch.manual_seed(hp.run_name)
random.seed(hp.run_name)
processor = DataProcessor(trainset_size=5 / 6, valset_size=1 / 6, batch_size=hp.batch_size)

if not hp.curriculum:
    processor.addPepperNoise(hp.noise_percent)
for runs in range(10):
    hp.run_name = runs
    running_noise = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_path = "results/experiment_mnist/runs/sequence " + str(hp.sequence_name) + "/sequence " + str(
        hp.sequence_name) + "_run " + str(
        hp.run_name)

    visualizer = DataVisualizer(log_path)
    analyzer = StatisticalAnalyzer()
    if hp.curriculum:
        running_noise = 0
        processor.addPepperNoise(running_noise)
        postfix = "with curriculum.csv"
    else:
        running_noise = hp.noise_percent
        postfix = "without curriculum.csv"

    filename = "results/experiment_mnist/metrics/sequence " + str(hp.sequence_name) + "/sequence " + str(
        hp.sequence_name) + "_" + "run " + str(
        hp.run_name) + "_" + postfix
    sequence_result = "results/experiment_mnist/metrics/sequence " + str(
        hp.sequence_name) + "/summary of sequence " + str(
        hp.sequence_name) + ".csv"

    if not os.path.exists(filename):
        with open(filename, 'w+', newline='') as csvfile:
            fwrite = csv.writer(csvfile, delimiter=',')
            fwrite.writerow(["Sequence", "Run Name", "Learning Rate", "Batch Size", "Noise Percent",
                             "Curriculum", "Epoch", "Train Loss", "Train Accuracy", "Train F1",
                             "Test Loss", "Test Accuracy", "Test F1", "Validation Accuracy", "Validation F1"])
    if not os.path.exists(sequence_result):
        with open(sequence_result, 'w+', newline='') as csvfile:
            fwrite = csv.writer(csvfile, delimiter=',')
            fwrite.writerow(
                ["Sequence", "repeat_id", "Run Name", "Learning Rate", "Batch Size", "Noise Percent", "Curriculum",
                 "Best Train Accuracy", "At Epoch", "Mean Train Accuracy",
                 "Best Test Accuracy", "At Epoch", "Mean test accuracy", "Test F1", "Validation Accuracy",
                 "Validation F1"])

    '''ResNet with 10 classes'''
    net = ResNet(hp.classes)
    if torch.cuda.is_available():
        net.cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=hp.learning_rate)

    ''' metrics array holds metrics of each epoch
        all_metrics holds all metrics from all epochs
        at_epoch_test holds the index of the epoch that the
        best test accuracy occurred
    '''
    all_metrics = list()

    train_accuracies = torch.zeros((hp.epoches, 1))
    test_accuracies = torch.zeros((hp.epoches, 1))
    for epoch in range(hp.epoches):
        metrics = dict()
        metrics["validation_accuracy"] = 0
        metrics["validation_f1score"] = 0
        net.batch_size = hp.batch_size
        if hp.curriculum:
            running_noise += hp.noise_percent / hp.epoches
            processor.addPepperNoise(running_noise)
        confusion_matrix_train = torch.zeros((10, 10))
        datasets = [processor.train_set, processor.test_set]
        if epoch == hp.epoches - 1:
            datasets.append(processor.validation_set)
        for i, dataloader in enumerate(datasets):
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
                if dataloader == processor.train_set:
                    batch_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            train_accuracy = train_correct / len(dataloader.dataset)
            f1score_train = analyzer.fonescore(confusion_matrix_train)
            if dataloader == processor.train_set:
                metrics["train_loss"] = train_loss.item()
                metrics["train_accuracy"] = train_accuracy.item()
                metrics["train_f1score"] = f1score_train.item()
                metrics["train_confusion_matrix"] = confusion_matrix_train.numpy()
            elif dataloader == processor.test_set:
                metrics["test_loss"] = train_loss.item()
                metrics["test_accuracy"] = train_accuracy.item()
                metrics["test_f1score"] = f1score_train.item()
                metrics["test_confusion_matrix"] = confusion_matrix_train.numpy()
            else:
                metrics["validation_accuracy"] = train_accuracy.item()
                metrics["validation_f1score"] = f1score_train.item()

        fig = visualizer.plotMatrix([metrics["train_confusion_matrix"], metrics["test_confusion_matrix"]])

        visualizer.addToTensorboard(fig, epoch)
        plt.close("all")
        all_metrics.append(metrics)
        train_accuracies[epoch] = metrics["train_accuracy"]
        test_accuracies[epoch] = metrics["test_accuracy"]
        print(f'Epoch:{epoch},Run:{hp.run_name},current noise:{running_noise},'
              f'Train loss:{metrics["train_loss"]:.4f},Train accuracy:{metrics["train_accuracy"]:.4f},'
              f'train F1 score:{metrics["train_f1score"]:.4f},Test loss:{metrics["test_loss"]:.4f},'
              f'Test accuracy:{metrics["test_accuracy"]:.4f},Test f1 score:{metrics["test_f1score"]:.4f}')

        with open(filename, 'a', newline='') as csvfile:
            fwrite = csv.writer(csvfile, delimiter=',')
            fwrite.writerow(
                [hp.sequence_name, hp.run_name, hp.learning_rate, hp.batch_size, running_noise, hp.curriculum, epoch,
                 metrics["train_loss"],
                 metrics["train_accuracy"],
                 metrics["train_f1score"],
                 metrics["test_loss"], metrics["test_accuracy"], metrics["test_f1score"],
                 metrics["validation_accuracy"], metrics["validation_f1score"]])
    at_epoch_train = torch.argmax(train_accuracies).item()
    at_epoch_test = torch.argmax(test_accuracies).item()
    mean_train_accuracy = torch.mean(train_accuracies).item()
    mean_test_accuracy = torch.mean(test_accuracies).item()
    with open(sequence_result, 'a', newline='') as csvfile:
        fwrite = csv.writer(csvfile, delimiter=',')
        fwrite.writerow(
            [hp.sequence_name, runs, hp.run_name, hp.learning_rate, hp.batch_size, running_noise, hp.curriculum,
             all_metrics[at_epoch_train]["train_accuracy"], at_epoch_train,
             mean_train_accuracy, all_metrics[at_epoch_test]["test_accuracy"], str(at_epoch_test),
             mean_test_accuracy, all_metrics[at_epoch_test]["test_f1score"], metrics["validation_accuracy"],
             metrics["validation_f1score"]])
