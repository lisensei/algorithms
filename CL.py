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
from sklearn.metrics import f1_score

# Hyper parameters
parser = argparse.ArgumentParser(description="Hyper parameters")
parser.add_argument("-rn", default=0, help="run name")
parser.add_argument("-lr", default=3e-3, help="learning rate")
parser.add_argument("-ne", default=20, help="number of epoch")
parser.add_argument("-bs", default=16, help="batch size")
parser.add_argument("-mean", default=0, help="Gaussian nose mean")
parser.add_argument("-std", default=1, help="Gaussian nose std")
parser.add_argument("-classes", default=10, help="number of classes")
hp = parser.parse_args()
runname = hp.rn
lr = hp.lr
batchSize = hp.bs
epoch = hp.ne
mean = hp.mean
std = hp.std
number_of_class = hp.classes
filename = str(runname) + "metrics_with_curriculum_cl.csv"
with open(filename, 'a', newline='') as csvfile:
    fwrite = csv.writer(csvfile, delimiter=',')
    fwrite.writerow(["Epoch", "Train Loss", "Test Loss", "Train Accuracy",
                     "Test Accuracy", "Train F1", "Test F1",
                     "Train Confusion Matrix", "Test Confusition Matrix"])


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
        torchvision.transforms.Normalize(mean=mean, std=std)
    ]
), download=True)
train_set = Data.DataLoader(dataset=Data.TensorDataset(data_train.data.to(torch.float32).unsqueeze(dim=1),
                                                       data_train.targets), batch_size=batchSize, shuffle=True)
test_set = Data.DataLoader(dataset=Data.TensorDataset(data_test.data.to(torch.float32).unsqueeze(dim=1),
                                                      data_test.targets), batch_size=batchSize, shuffle=False)


class ResBlock(nn.Module):
    def __init__(self, in_channels, middle_out, out_channels):
        super(ResBlock, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=middle_out, kernel_size=3, padding=1)
        self.r1 = nn.ReLU();
        self.cnn2 = nn.Conv2d(in_channels=middle_out, out_channels=out_channels, kernel_size=3, padding=1)
        self.r2 = nn.ReLU();

    def forward(self, x):
        return x


class CNN(nn.Module):
    def __init__(self, number_of_class):
        self.batchSize = batchSize
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.r1 = nn.ReLU();
        self.cnn2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        self.r2 = nn.ReLU();
        self.cnn3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.r3 = nn.ReLU();
        self.cnn4 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.r4 = nn.ReLU();
        self.fc = nn.Linear(in_features=28 * 28 * 4, out_features=number_of_class)

    def forward(self, x):
        out = self.cnn1(x);
        out = self.r1(out)
        out = self.cnn2(out);
        z1 = self.r2(out)
        out = self.cnn3(z1 + x);
        out = self.r3(out)
        out = self.cnn4(out + z1);
        out = self.r4(out)
        out = out.reshape(self.batchSize, 28 * 28 * 4)
        out = self.fc(out)
        return out


net = CNN(number_of_class)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
is_curriculum_enabled = False
if not is_curriculum_enabled:
    curriculum_noise.percent_noise = 50
print("Learning rate set to", lr, "Batch size set to", batchSize)
'''Training with curriculum:'''
print("\n")
print("Training with curriculum starts:")

for iteration in range(epoch):
    train_loss = 0
    test_loss = 0
    train_correct = 0
    net.batchSize = batchSize
    # Train F1 variables and confusion matrix declaration
    f1score_ypredict = torch.zeros((10, 1))
    f1score_y = torch.zeros((10, 1))
    confusion_matrix_train = torch.zeros((10, 10))

    for j, (x, y) in enumerate(train_set):
        x = x.to(device)
        y = y.to(device)
        ypredict = net.forward(x)
        ypredict = ypredict.to(torch.float32)
        temp = torch.argmax(ypredict, dim=1)
        for k, v in zip(y, temp):
            f1score_ypredict[v] += 1
            f1score_y[k] += 1
            confusion_matrix_train[k, v] += 1
        train_correct += ((temp - y) == 0).sum()
        l = loss(ypredict, y)
        train_loss += l
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        if j % 1000 == 0:
            print("epoch:", iteration, "batch:", j, "Loss:", l.item())
    train_accuracy = train_correct / 60000
    if is_curriculum_enabled:
        if epoch % 10 == 0:
            curriculum_noise.percent_noise += 10
    # Calculate Train F1
    y1 = f1score_y.to(torch.int32).flatten().tolist()
    y2 = f1score_ypredict.to(torch.int32).flatten().tolist()
    f1 = f1_score(y1, y2, average='macro')

    # Test F1 variables and confusion matrix declaration
    f1score_ypredict_test = torch.zeros((10, 1))
    f1score_y_test = torch.zeros((10, 1))
    confusion_matrix_test = torch.zeros((10, 10))
    test_correct = 0
    with torch.no_grad():
        for i in range(1):
            for j, (x, y) in enumerate(test_set):
                x = x.to(device)
                y = y.to(device)
                net.batchSize = batchSize
                ypredict = net(x)
                l = loss(ypredict, y)
                test_loss += l
                temp = torch.argmax(ypredict, dim=1)
                for k, v in zip(y, temp):
                    f1score_ypredict_test[v] += 1
                    f1score_y_test[k] += 1
                    confusion_matrix_test[k, v] += 1
                test_correct += ((temp - y) == 0).sum()

    y1_test = f1score_y_test.to(torch.int32).flatten().tolist()
    y2_test = f1score_ypredict_test.to(torch.int32).flatten().tolist()
    f1score_test = f1_score(y1_test, y2_test, average='macro')
    test_accuracy = test_correct / 10000
    print(
        f'Train loss:{train_loss.item():.4f}, Train accuracy:{train_accuracy.item():.4f},Test loss:{test_loss.item():.4f},Test accuracy:{test_accuracy.item():.4f},Train F1 score:{f1:.4f},Test F1 score:{f1score_test:.4f}')

    with open(filename, 'a', newline='') as csvfile:
        fwrite = csv.writer(csvfile, delimiter=',')
        fwrite.writerow(
            [iteration, train_loss.item(), test_loss.item(), train_accuracy.item(), test_accuracy.item(), f1,
             f1score_test,
             confusion_matrix_train,
             confusion_matrix_test])
