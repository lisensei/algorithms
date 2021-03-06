import argparse
import csv
import os
import random
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/gdrive')
'''Argparse hyper parameters'''
os.chdir('/gdrive/My Drive/CL/')
parser = argparse.ArgumentParser(description="Hyper parameters")
parser.add_argument("-run_name", default=0)
parser.add_argument("-sequence_name", default=19)
parser.add_argument("-learning_rate", default=3e-3)
parser.add_argument("-epoches", default=20)
parser.add_argument("-batch_size", default=16)
parser.add_argument("-noise_percent", default=19)
parser.add_argument("-mean", default=0)
parser.add_argument("-std", default=1)
parser.add_argument("-classes", default=10)
parser.add_argument("-curriculum", default=False)
parser.add_argument("-samples", default=1000)
hp = parser.parse_args(args=[])

'''This class is used to send data to tensorboard'''


class DataVisualizer:
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def addToTensorboard(self, figure, step):
        self.writer.add_figure(tag="confusion matrix", figure=figure, global_step=step)

    def plotMatrix(self, matrices, show_figure=False):
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
            if show_figure:
                plt.show()
        return figure


'''This class is used to process dataset,such as adding noise'''


class DataProcessor:
    def __init__(self, samples):
        self.samples = samples
        self.data_train = torchvision.datasets.MNIST(root='/data', train=True, transform=transform.Compose(
            [
                transform.ToTensor()
            ]
        ), download=True)

        self.data_test = torchvision.datasets.MNIST(root='/data', train=False, transform=transform.Compose(
            [
                transform.ToTensor(),
            ]
        ), download=True)
        self.train_data = torch.clone(self.data_train.data[0:samples])
        self.train_target = torch.clone(self.data_train.targets[0:samples])
        self.train_set = Data.DataLoader(
            dataset=Data.TensorDataset(self.train_data.to(torch.float32).unsqueeze(dim=1),
                                       self.train_target), batch_size=hp.batch_size, shuffle=True)
        self.test_set = Data.DataLoader(
            dataset=Data.TensorDataset(self.data_test.data.to(torch.float32).unsqueeze(dim=1),
                                       self.data_test.targets), batch_size=hp.batch_size, shuffle=False)

    def repack(self):
        self.train_set = Data.DataLoader(
            dataset=Data.TensorDataset(self.train_data.to(torch.float32).unsqueeze(dim=1),
                                       self.train_target), batch_size=hp.batch_size, shuffle=True)
        self.test_set = Data.DataLoader(
            dataset=Data.TensorDataset(self.data_test.data.to(torch.float32).unsqueeze(dim=1),
                                       self.data_test.targets), batch_size=hp.batch_size, shuffle=False)

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
        self.train_data = torch.clone(self.data_train.data[0:self.samples])
        self.train_target = torch.clone(self.data_train.targets[0:self.samples])


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
        out = out.reshape(len(out), 28 * 28 * 8)
        out = self.linear_layer.forward(out)
        return out


processor = DataProcessor(hp.samples)
if not hp.curriculum:
    processor.addPepperNoise(hp.noise_percent)
for runs in range(30):
    hp.run_name = runs
    running_noise = 0
    torch.manual_seed(hp.run_name)
    random.seed(hp.run_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_path = "runs/mnist/sequence " + str(hp.sequence_name) + "/sequence " + str(hp.sequence_name) + "_run " + str(
        hp.run_name)

    visualizer = DataVisualizer(log_path)
    analyzer = StatisticalAnalyzer()
    '''define necessary variables and file names'''
    if hp.curriculum:
        running_noise = 0
        processor.addPepperNoise(running_noise)
        postfix = "with curriculum.csv"
    else:
        running_noise = hp.noise_percent
        postfix = "without curriculum.csv"
    '''Sequence file and sequences file'''
    filename = "/gdrive/My Drive/CL/metrics/sequence " + str(hp.sequence_name) + "/sequence " + str(
        hp.sequence_name) + "_" + "run " + str(
        hp.run_name) + "_" + postfix
    sequence_result = "/gdrive/My Drive/CL/metrics/sequence " + str(hp.sequence_name) + "/summary of sequence " + str(
        hp.sequence_name) + ".csv"
    if not os.path.exists(filename):
        with open(filename, 'a', newline='') as csvfile:
            fwrite = csv.writer(csvfile, delimiter=',')
            fwrite.writerow(["Sequence", "Run Name", "Learning Rate", "Batch Size", "Noise Percent",
                             "Curriculum", "Epoch", "Train Loss", "Train Accuracy", "Train F1",
                             "Test Loss", "Test Accuracy", "Test F1"])

    if not os.path.exists(sequence_result):
        with open(sequence_result, 'a', newline='') as csvfile:
            fwrite = csv.writer(csvfile, delimiter=',')
            fwrite.writerow(["Sequence", "Run Name", "Learning Rate", "Batch Size", "Noise Percent", "Curriculum",
                             "Best Train Accuracy", "Mean Train Accuracy", "Best Test Loss", "Best Test Accuracy",
                             "Mean test accuracy", "Test F1"])

    '''ResNet with 10 classes'''
    net = ResNet(hp.classes)
    if torch.cuda.is_available():
        net.cuda()
        torch.cuda.manual_seed(hp.run_name)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=hp.learning_rate)

    ''' metrics array holds metrics of each epoch
        all_metrics holds all metrics from all epochs
        at_epoch holds the index of the epoch that the
        best test accuracy occurred
    '''
    all_metrics = list()
    train_accuracies = torch.zeros((hp.epoches, 1))
    test_accuracies = torch.zeros((hp.epoches, 1))
    for epoch in range(hp.epoches):
        metrics = dict()
        net.batch_size = hp.batch_size
        if hp.curriculum:
            running_noise += hp.noise_percent / hp.epoches
            processor.addPepperNoise(running_noise)
        confusion_matrix_train = torch.zeros((10, 10))
        for i, dataloader in enumerate([processor.train_set, processor.test_set]):
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
            else:
                metrics["test_loss"] = train_loss.item()
                metrics["test_accuracy"] = train_accuracy.item()
                metrics["test_f1score"] = f1score_train.item()
                metrics["test_confusion_matrix"] = confusion_matrix_train.numpy()
        fig = visualizer.plotMatrix([metrics["train_confusion_matrix"], metrics["test_confusion_matrix"]],
                                    show_figure=False)

        visualizer.addToTensorboard(fig, epoch)

        all_metrics.append(metrics)
        train_accuracies[epoch] = metrics["train_accuracy"]
        test_accuracies[epoch] = metrics["test_accuracy"]
        print(f'Epoch:{epoch}, Run:{hp.run_name},Current noise Percent:{running_noise}\n'
              f'Train loss:{metrics["train_loss"]:.4f}, Train accuracy:{metrics["train_accuracy"]:.4f},'
              f'train F1 score:{metrics["train_f1score"]:.4f}\nTest loss:{metrics["test_loss"]:.4f},'
              f'Test accuracy:{metrics["test_accuracy"]:.4f},Test f1 score:{metrics["test_f1score"]:.4f}')

        with open(filename, 'a', newline='') as csvfile:
            fwrite = csv.writer(csvfile, delimiter=',')
            fwrite.writerow(
                [hp.sequence_name, hp.run_name, hp.learning_rate, hp.batch_size, running_noise, hp.curriculum, epoch,
                 metrics["train_loss"],
                 metrics["train_accuracy"],
                 metrics["train_f1score"],
                 metrics["test_loss"], metrics["test_accuracy"], metrics["test_f1score"]])
    at_epoch_train = torch.argmax(train_accuracies).item()
    at_epoch_test = torch.argmax(test_accuracies).item()
    mean_train_accuracy = torch.mean(train_accuracies).item()
    mean_test_accuracy = torch.mean(test_accuracies).item()
    with open(sequence_result, 'a', newline='') as csvfile:
        fwrite = csv.writer(csvfile, delimiter=',')
        fwrite.writerow(
            [hp.sequence_name, hp.run_name, hp.learning_rate, hp.batch_size, running_noise, hp.curriculum,
             str(all_metrics[at_epoch_train]["train_accuracy"]) + " at epoch:" + str(at_epoch_train),
             mean_train_accuracy,
             all_metrics[at_epoch_test]["test_loss"],
             str(all_metrics[at_epoch_test]["test_accuracy"]) + " at epoch:" + str(at_epoch_test),
             mean_test_accuracy, all_metrics[at_epoch_test]["test_f1score"]])
