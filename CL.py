import argparse
import csv
import random

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.utils
import torch.utils
import torch.utils
import torch.utils.data as Data
import torchvision.models
import torchvision.transforms as transform

'''Argparse hyper parameters'''
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
parser.add_argument("-curriculum", default=False)
hp = parser.parse_args()

if hp.curriculum:
    postfix = "with curriculum.csv"
else:
    postfix = "without curriculum.csv"
'''Sequence file and sequences file'''
filename = "sequence " + str(hp.sequence_name) + "_" + "run " + str(
    hp.run_name) + "_" + postfix
sequence_result = "summary of sequence " + str(hp.sequence_name) + ".csv"
with open(filename, 'a', newline='') as csvfile:
    fwrite = csv.writer(csvfile, delimiter=',')
    fwrite.writerow(["Sequence", "Run Name", "Learning Rate", "Batch Size", "Noise Percent",
                     "Epoch", "Curriculum", "Train Loss", "Train Accuracy", "Train F1",
                     "Test Loss", "Test Accuracy", "Test F1",
                     "Train Confusion Matrix", "Test Confusition Matrix"])

with open(sequence_result, 'a', newline='') as csvfile:
    fwrite = csv.writer(csvfile, delimiter=',')
    fwrite.writerow(["Sequence", "Run Name", "Learning Rate", "Batch Size", "Noise Percent", "Curriculum",
                     "At Epoch", "Best Test Loss", "Best Test Accuracy", "Test F1"])

'''Pepper Noise Function'''


def addPepperNoise(source, percent):
    source = source.squeeze(dim=1)
    if percent >= 100:
        percent = 100

    for i, image in enumerate(source):
        polluted_pixels = random.sample(range(0, 28 * 28), int(percent / 100 * 28 * 28))
        for i in range(len(polluted_pixels)):
            row = int(polluted_pixels[i] / 28)
            col = polluted_pixels[i] % 28
            if (row + col) % 2 == 0:
                image[row][col] = 0
            else:
                image[row][col] = 255.0
    return source


def applyNoise():
    global training_data
    global training_target
    global train_set
    global test_set
    training_data = torch.clone(data_train.data)
    training_target = torch.clone(data_train.targets)
    training_data = addPepperNoise(training_data, curriculum_noise.percent_noise)
    train_set = Data.DataLoader(
        dataset=Data.TensorDataset(training_data.to(torch.float32).unsqueeze(dim=1),
                                   training_target), batch_size=hp.batch_size, shuffle=True)
    test_set = Data.DataLoader(
        dataset=Data.TensorDataset(data_test.data.to(torch.float32).unsqueeze(dim=1),
                                   data_test.targets), batch_size=hp.batch_size, shuffle=False)
    print("Noise applied")


class TransformNoise:
    def __init__(self):
        self.percent_noise = 0.0

    def __call__(self, x):
        return addPepperNoise(x, self.percent_noise)

    def __repr__(self):
        return self.__class__.__name__ + '()'


'''F1 Score function'''


def fonescore(matrix):
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


'''Data processing'''
curriculum_noise = TransformNoise()
device = "cuda" if torch.cuda.is_available() else "cpu"
data_train = torchvision.datasets.MNIST(root='/data', train=True, transform=transform.Compose(
    [
        transform.ToTensor(),
    ]
), download=True)

data_test = torchvision.datasets.MNIST(root='/data', train=False, transform=transform.Compose(
    [
        transform.ToTensor(),
    ]
), download=True)

training_data = torch.clone(data_train.data)
training_target = torch.clone(data_train.targets)
train_set = Data.DataLoader(
    dataset=Data.TensorDataset(training_data.to(torch.float32).unsqueeze(dim=1),
                               training_target), batch_size=hp.batch_size, shuffle=True)
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


'''ResNet with 10 classes'''
net = ResNet(hp.classes)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=hp.learning_rate)
if hp.curriculum:
    curriculum_noise.percent_noise = 0
else:
    curriculum_noise.percent_noise = 5
    applyNoise()
''' metrics array holds metrics of each epoch
    all_metrics holds all metrics from all epochs
    at_epoch holds the index of the epoch that the
    best test accuracy occurred
'''
all_metrics = list()
test_accuracies = torch.zeros((hp.epoches, 1))
for epoch in range(hp.epoches):
    metrics = []
    net.batch_size = hp.batch_size
    if hp.curriculum:
        curriculum_noise.percent_noise += 2
        applyNoise()
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
                print("epoch:", epoch, "batch:", j, "Loss:", batch_loss.item())
        train_accuracy = train_correct / len(dataloader.dataset)
        f1score_train = fonescore(confusion_matrix_train)
        metrics.append(train_loss)
        metrics.append(train_accuracy)
        metrics.append(f1score_train)
        metrics.append(confusion_matrix_train)

    all_metrics.append(metrics)
    test_accuracies[epoch] = metrics[5]
    print(
        f'Train loss:{metrics[0]:.4f}, Train accuracy:{metrics[1]:.4f},'
        f'train F1 score:{metrics[2]:.4f},Test loss:{metrics[4]:.4f},'
        f'Test accuracy:{metrics[5]:.4f},Test f1 score:{metrics[6]:.4f}')

    with open(filename, 'a', newline='') as csvfile:
        fwrite = csv.writer(csvfile, delimiter=',')
        fwrite.writerow(
            [hp.sequence_name, hp.run_name, hp.learning_rate, hp.batch_size, hp.noise_percent, hp.curriculum, epoch,
             metrics[0],
             metrics[1],
             metrics[2], metrics[3],
             metrics[4], metrics[5], metrics[6], metrics[7]])
at_epoch = torch.argmax(test_accuracies)
with open(sequence_result, 'a', newline='') as csvfile:
    fwrite = csv.writer(csvfile, delimiter=',')
    fwrite.writerow([hp.sequence_name, hp.run_name, hp.learning_rate, hp.batch_size, hp.noise_percent, hp.curriculum,
                     at_epoch, all_metrics[at_epoch][4], all_metrics[at_epoch][5], all_metrics[at_epoch][6]])
