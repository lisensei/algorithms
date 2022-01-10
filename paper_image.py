import numpy.random
import torch
import torchvision.datasets as datasets
import torch.utils.data as utils
import scipy.stats as st
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import numpy as np
import os

np.random.seed(424242)


def add_pepper(x, percent):
    for i, image in enumerate(x):
        if torch.max(image) > 1:
            image = image / 255
        polluted_pixels = random.sample(range(0, image.shape[1] ** 2), int(percent / 100 * image.shape[1] ** 2))
        for chn in range(image.shape[0]):
            for i in range(len(polluted_pixels)):
                row = int(polluted_pixels[i] / image.shape[1])
                col = polluted_pixels[i] % image.shape[1]
                if (row + col) % 2 == 0:
                    image[chn][row][col] = 0
                else:
                    image[chn][row][col] = 1


def generated_pictures(samples_dic):
    for k, v in samples_dic.items():
        for i, (x, y) in enumerate(v):
            plt.subplot(2, 5, i + 1)
            plt.axis("off")
            plt.imshow(torch.permute(x[0], (1, 2, 0)))
            plt.subplot(2, 5, 6 + i)
            add_pepper(x, 20)
            plt.axis("off")
            plt.imshow(torch.permute(x[0], (1, 2, 0)))
            fname = k + "_clean_and_20 percent_noise.png"
            plt.savefig(fname)


def noise_curriculum_pictures(samples_dic, noise_levels):
    for num in noise_levels:
        for k, v in samples_dic.items():
            for i, (x, y) in enumerate(v):
                plt.subplot(2, 5, i + 1)
                plt.axis("off")
                y = torch.clone(x)
                add_pepper(y, num)
                plt.imshow(torch.permute(y[0], (1, 2, 0)))
                plt.subplot(2, 5, 6 + i)
                add_pepper(x, 20)
                plt.axis("off")
                plt.imshow(torch.permute(x[0], (1, 2, 0)))
                fname = k + "_noise curriculum_" + str(num) + " percent_noise.png"
                plt.savefig(fname)


size = 5
mnist_samples = datasets.MNIST(root="./data", transform=transforms.ToTensor())
kmnist_samples = datasets.KMNIST(root="./data", transform=transforms.ToTensor())
cifar10_samples = datasets.CIFAR10(root="./data", transform=transforms.ToTensor())
mnist_indices = st.randint.rvs(0, len(mnist_samples), size=size)
kmnist_indices = st.randint.rvs(0, len(kmnist_samples), size=size)
cifar10_indices = st.randint.rvs(0, len(cifar10_samples), size=size)

mnist_images = utils.DataLoader(utils.Subset(mnist_samples, mnist_indices))
kmnist_images = utils.DataLoader(utils.Subset(kmnist_samples, kmnist_indices))
cifar10_images = utils.DataLoader(utils.Subset(cifar10_samples, cifar10_indices))
plt.axis("off")
samples_dic = {"mnist": mnist_images, "kmnist": kmnist_images, "cifar10": cifar10_images}

path = "C:/Users/sherlock/Desktop"
os.chdir(path)
noise_levels = [1, 10, 20]
noise_curriculum_pictures({"cifar10": samples_dic["cifar10"]}, noise_levels)
