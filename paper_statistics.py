import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import torch

np.set_printoptions(precision=6)
import os


class summarizer():
    def __init__(self, filepath, experiment):
        self.path = filepath
        self.experiment = experiment
        self.repeats = 10
        self.rows = 10
        self.sequences = 40

    def get_loss(self):
        mean_trainloss_nc = np.zeros((self.sequences, 1))
        mean_trainloss_cl = np.zeros((self.sequences, 1))
        losstable_nc = np.zeros((self.sequences, self.repeats))
        losstable_cl = np.zeros((self.sequences, self.repeats))
        for i in range(1, self.sequences + 1):
            for j in range(0, 10):
                filename = filepath + "sequence " + str(i) + "/sequence " + str(i) + "_run " + str(
                    j) + "_with curriculum.csv"
                mean_loss = np.mean(pd.read_csv(filename, usecols=[7]).to_numpy(), axis=0)
                losstable_cl[i - 1, j] = mean_loss
        for i in range(1, self.sequences + 1):
            for j in range(0, 10):
                filename = filepath + "sequence " + str(i) + "/sequence " + str(i) + "_run " + str(
                    j) + "_without curriculum.csv"
                mean_loss = np.mean(pd.read_csv(filename, usecols=[7]).to_numpy(), axis=0)
                losstable_nc[i - 1, j] = mean_loss
        mean_trainloss_nc = np.mean(losstable_nc, axis=1)
        mean_trainloss_cl = np.mean(losstable_cl, axis=1)
        xs = np.linspace(1, 40, 40)
        plt.plot(xs, mean_trainloss_nc)
        plt.plot(xs, mean_trainloss_cl)
        plt.title("mean training loss of " + self.experiment + " experiment")
        plt.xlabel("percent of pepper noise")
        plt.ylabel("mean training loss")
        plt.legend(["without curriculum", "with curriculum"])
        fname = "figures/experiment_" + self.experiment + "/" + self.experiment + "_mean_training_loss.png"
        plt.savefig(fname.lower())
        plt.clf()
        return (mean_trainloss_nc, mean_trainloss_cl)

    def get_validation(self):
        valtable_nc = np.zeros((self.sequences, 1))
        valtable_cl = np.zeros((self.sequences, 1))
        for i in range(1, 41):
            filename = filepath + "sequence " + str(i) + "/summary of sequence " + str(i) + ".csv"
            valtable_nc[i - 1] = np.mean(pd.read_csv(filename, usecols=[14]).to_numpy()[:10], axis=0)
            valtable_cl[i - 1] = np.mean(pd.read_csv(filename, usecols=[14]).to_numpy()[10:], axis=0)
        xs = np.linspace(1, self.sequences, self.sequences)
        plt.plot(xs, valtable_nc)
        plt.plot(xs, valtable_cl)
        plt.title("mean validation accuracy of " + self.experiment + " experiment")
        plt.xlabel("percent of pepper noise")
        plt.ylabel("mean validation accuracy")
        plt.legend(["without curriculum", "with curriculum"])
        fname = "figures/experiment_" + self.experiment + "/" + self.experiment + "_mean_validation_accuracy.png"
        plt.savefig(fname.lower())
        plt.clf()
        return (valtable_nc, valtable_cl)

    def get_best_test(self):
        best_test_nc = np.zeros((self.sequences, 1))
        best_test_cl = np.zeros((self.sequences, 1))
        accholder_nc = np.zeros((self.repeats, 1))
        accholder_cl = np.zeros((self.repeats, 1))
        for i in range(1, self.sequences + 1):
            for j in range(0, 10):
                filename_nc = filepath + "sequence " + str(i) + "/sequence " + str(i) + "_run " + str(
                    j) + "_without curriculum.csv"
                filename_cl = filepath + "sequence " + str(i) + "/sequence " + str(i) + "_run " + str(
                    j) + "_with curriculum.csv"
                accholder_nc[j] = np.max(pd.read_csv(filename_nc, usecols=[11]).to_numpy())
                accholder_cl[j] = np.max(pd.read_csv(filename_cl, usecols=[11]).to_numpy())
            best_test_nc[i - 1] = np.max(accholder_nc)
            best_test_cl[i - 1] = np.max(accholder_cl)
        xs = np.linspace(1, self.sequences, self.sequences)
        plt.plot(xs, best_test_nc)
        plt.plot(xs, best_test_cl)
        plt.title("best test accuracy of " + self.experiment + " experiment")
        plt.xlabel("percent of pepper noise")
        plt.ylabel("best test accuracy")
        plt.legend(["without curriculum", "with curriculum"])
        fname = "figures/experiment_" + self.experiment + "/" + self.experiment + "_best_test_accuracy.png"
        plt.savefig(fname.lower())
        plt.clf()
        return (best_test_nc, best_test_cl)

    def get_mean_train_accuracy(self):
        meantrainacc_nc = np.zeros((self.sequences, 1))
        meantrainacc_cl = np.zeros((self.sequences, 1))
        meanholder_nc = np.zeros((self.repeats, 1))
        meanholder_cl = np.zeros((self.repeats, 1))
        for i in range(1, self.sequences + 1):
            for j in range(0, 10):
                filename_nc = filepath + "sequence " + str(i) + "/sequence " + str(i) + "_run " + str(
                    j) + "_without curriculum.csv"
                filename_cl = filepath + "sequence " + str(i) + "/sequence " + str(i) + "_run " + str(
                    j) + "_with curriculum.csv"
                meanholder_nc[j] = np.mean(pd.read_csv(filename_nc, usecols=[8]).to_numpy())
                meanholder_cl[j] = np.mean(pd.read_csv(filename_cl, usecols=[8]).to_numpy())
            meantrainacc_nc[i - 1] = np.mean(meanholder_nc)
            meantrainacc_cl[i - 1] = np.mean(meanholder_cl)
        xs = np.linspace(1, self.sequences, self.sequences)
        plt.plot(xs, meantrainacc_nc)
        plt.plot(xs, meantrainacc_cl)
        plt.title("mean training accuracy of " + self.experiment + " experiment")
        plt.xlabel("percent of pepper noise")
        plt.ylabel("mean training accuracy")
        plt.legend(["without curriculum", "with curriculum"])
        fname = "figures/experiment_" + self.experiment + "/" + self.experiment + "_mean_training_accuracy.png"
        plt.savefig(fname.lower())
        plt.clf()
        return (meantrainacc_nc, meantrainacc_cl)

    def get_mean_test(self):
        meantestacc_nc = np.zeros((self.sequences, 1))
        meantestacc_cl = np.zeros((self.sequences, 1))
        accholder_nc = np.zeros((self.repeats, 1))
        accholder_cl = np.zeros((self.repeats, 1))
        for i in range(1, self.sequences + 1):
            for j in range(0, 10):
                filename_nc = filepath + "sequence " + str(i) + "/sequence " + str(i) + "_run " + str(
                    j) + "_without curriculum.csv"
                filename_cl = filepath + "sequence " + str(i) + "/sequence " + str(i) + "_run " + str(
                    j) + "_with curriculum.csv"
                accholder_nc[j] = np.mean(pd.read_csv(filename_nc, usecols=[11]).to_numpy())
                accholder_cl[j] = np.mean(pd.read_csv(filename_cl, usecols=[11]).to_numpy())
            meantestacc_nc[i - 1] = np.mean(accholder_nc)
            meantestacc_cl[i - 1] = np.mean(accholder_cl)
        xs = np.linspace(1, self.sequences, self.sequences)
        plt.plot(xs, meantestacc_nc)
        plt.plot(xs, meantestacc_cl)
        plt.title("mean test accuracy of " + self.experiment + " experiment")
        plt.xlabel("percent of pepper noise")
        plt.ylabel("mean test accuracy")
        plt.legend(["without curriculum", "with curriculum"])
        fname = "figures/experiment_" + self.experiment + "/" + self.experiment + "_mean_test_accuracy.png"
        plt.savefig(fname.lower())
        plt.clf()
        return meantestacc_nc, meantestacc_cl

    def get_raw_test(self):
        raw_testacc_nc = np.zeros((self.sequences, self.repeats))
        raw_testacc_cl = np.zeros((self.sequences, self.repeats))
        for i in range(1, self.sequences + 1):
            for j in range(0, 10):
                filename_nc = filepath + "sequence " + str(i) + "/sequence " + str(i) + "_run " + str(
                    j) + "_without curriculum.csv"
                filename_cl = filepath + "sequence " + str(i) + "/sequence " + str(i) + "_run " + str(
                    j) + "_with curriculum.csv"
                raw_testacc_nc[i - 1, j] = np.mean(pd.read_csv(filename_nc, usecols=[11]).to_numpy())
                raw_testacc_cl[i - 1, j] = np.mean(pd.read_csv(filename_cl, usecols=[11]).to_numpy())

        return raw_testacc_nc, raw_testacc_cl

    def get_tablecsv(self):
        cl_table = np.zeros((40, 3))
        nc_table = np.zeros((40, 3))
        for i in range(1, 41):
            filename = filepath + "sequence " + str(i) + "/summary of sequence " + str(i) + ".csv"
            summary_nc = np.mean(pd.read_csv(filename, usecols=[9, 12, 14]).to_numpy()[:self.rows], axis=0)
            summary_cl = np.mean(pd.read_csv(filename, usecols=[9, 12, 14]).to_numpy()[self.rows:], axis=0)
            cl_table[i - 1] = np.round(summary_cl, decimals=6)
            nc_table[i - 1] = np.round(summary_nc, decimals=6)
        sequence_losses_nc = np.array(self.get_loss()[0]).reshape(-1, 1)
        best_test_nc = np.array(self.get_best_test()[0]).reshape(-1, 1)
        sequence_losses_cl = np.array(self.get_loss()[1]).reshape(-1, 1)
        best_test_cl = np.array(self.get_best_test()[1]).reshape(-1, 1)
        nc_table = np.concatenate(
            (np.arange(1, self.sequences + 1).reshape(-1, 1), sequence_losses_nc, best_test_nc, nc_table), axis=1)
        cl_table = np.concatenate(
            (np.arange(1, self.sequences + 1).reshape(-1, 1), sequence_losses_cl, best_test_cl, cl_table), axis=1)
        df_nc = pd.DataFrame(nc_table, columns=["Noise Percent", "Mean Training Loss", "Best Test Accuracy",
                                                "Mean Training Accuracy", "Mean Test Accuracy",
                                                "Mean Validation Accuracy"])
        df_cl = pd.DataFrame(cl_table, columns=["Noise Percent", "Mean Training Loss", "Best Test Accuracy",
                                                "Mean Training Accuracy", "Mean Test Accuracy",
                                                "Mean Validation Accuracy"])
        tablename_nc = "./tables/" + self.experiment + "_summary_nc.csv"
        tablename_cl = "./tables/" + self.experiment + "_summary_cl.csv"
        df_nc.to_csv(tablename_nc, index=False)
        df_cl.to_csv(tablename_cl, index=False)

    def get_confidence_interval(self, noise_level):
        metrics = dict()
        val_acc_nc = np.zeros((1, self.repeats))
        val_acc_cl = np.zeros((1, self.repeats))
        for i in range(self.repeats):
            filename_nc = filepath + "sequence " + str(noise_level) + "/sequence " + str(noise_level) + "_run " + str(
                i) + "_without curriculum.csv"
            val_acc_nc[0, i] = pd.read_csv(filename_nc, usecols=[13]).to_numpy().reshape(-1)[-1]
            filename_cl = filepath + "sequence " + str(noise_level) + "/sequence " + str(noise_level) + "_run " + str(
                i) + "_with curriculum.csv"
            val_acc_cl[0, i] = pd.read_csv(filename_cl, usecols=[13]).to_numpy()[-1]
        difference_table = pd.DataFrame(np.round(np.concatenate((val_acc_cl, val_acc_nc), axis=0), decimals=6),
                                        columns=["run 0", "run 1", "run 2", "run 3", "run 4",
                                                 "run 5", "run 6", "run 7", "run 8", "run 9", ])
        difference_table.index = ["curriculum", "non-curriculum"]
        # difference_table.to_csv(self.experiment + "_group_differences.csv")
        diffrence = (val_acc_cl - val_acc_nc).reshape(-1)
        diffrence_mean = np.sum(diffrence) / self.repeats
        diffrence_variance = np.sum((diffrence - diffrence_mean) ** 2 / (self.repeats - 1))
        std = np.sqrt(diffrence_variance)
        half_width = 2.262 * std / (self.repeats)
        lower = diffrence_mean - half_width
        upper = diffrence_mean + half_width
        metrics["Mean"] = np.round(diffrence_mean, decimals=4)
        metrics["Std"] = np.round(std, decimals=4)
        metrics["C.I."] = "[" + str(np.round(lower, decimals=4)) + "," + str(
            np.round(upper, decimals=4)) + "]"
        return metrics


def process():
    processor = summarizer(filepath, experiment)
    processor.get_loss()
    processor.get_mean_train_accuracy()
    processor.get_validation()
    processor.get_mean_test()
    processor.get_best_test()


'''Paper Statistics
Using Bin(40,1/2) to approximate normal(20,1)
$D=(D_1+D_2+...+D_{10})/10 
where D_i = \sum_{j=1}^{40} p(x=j)d_j$ 
and $d_j$ represents
The difference of test accuracy between curriculum group and non-curriculum group
Under a j\% pepper noise.
'''

x = []
n=40
for i in range(1, n+1):
    x.append(st.binom.pmf(i, n, 1 / 2))

pmf = np.array(x)
experiments = ["MNIST", "KMNIST", "CIFAR10"]
df_final = pd.DataFrame()
for experiment in experiments:
    filepath = "results/experiment_" + experiment + "/metrics/"
    processor = summarizer(filepath, experiment)
    test_nc, test_cl = processor.get_raw_test()
    ds = test_cl - test_nc
    Ds = pmf.transpose() @ ds
    mean = np.mean(Ds)
    std = np.std(Ds, ddof=1)
    half_width = 2.262 * std / np.sqrt(processor.repeats)
    print(f"Mean:{mean},Std:{std},Confidence Interval:[{mean - half_width},{mean + half_width}]")
