import os
import time
import torch
from art.attacks.poisoning import PoisoningAttackBackdoor
from torch.utils.data import TensorDataset, DataLoader

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.utils import  preprocess

from server import cur_round
import client
from config import SAVE_PATH, device
from utils import get_data_by_indices
from client import Client
import logging
from random import random
from matplotlib.ticker import MaxNLocator
from art.attacks.evasion import BoundaryAttack, AdversarialPatchPyTorch
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image

import warnings

class Adv_Client(Client):
    def __init__(self, name, pr = 0.3):
        super(Adv_Client, self).__init__(name)
        self.acc_benign = []
        self.acc_poison = []
        self.poison_ind = []
        self.poison = None
        self.poison_rate = pr
        self.save_example = True

    def poison_data(self):
        pass


    def adv_metrics(self):
        self.model.eval()
        eval_benign = []
        eval_adversial = []
        for s, i in enumerate(self.poison_ind):
            x, y = self.dataloader.dataset[s * self.batch:s * self.batch + self.batch]
            x = x.to(device)
            y = y.to(device)
            self.model.to(device)

            pred: object = self.model(x).max(dim=1).indices
            if i == 0:
                eval_benign.append((torch.sum(pred == y) / len(y)).cpu())
            else:
                eval_adversial.append((torch.sum(pred == y) / len(y)).cpu())
        acc = np.mean(eval_benign)
        adv_acc = np.mean(eval_adversial)
        return acc, adv_acc

    def finish_function(self):
        benign_acc, adv_acc = self.adv_metrics()
        self.logger.info(f"Adv. metrics using final model: benign_acc: {benign_acc}, adv_acc: {adv_acc} ")

    def plots(self):
        # Plot performance
        fig, ax = plt.subplots()
        ax.plot(list(range(len(self.losses))), self.losses, color='blue')
        ax.set_xlabel("Global Rounds")
        ax.set_ylabel('Loss')
        ax.legend(["Loss"], loc="center left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid()

        ax2 = ax.twinx()
        ax2.plot(list(range(len(self.losses))), self.accs, color='orange')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([-0.05, 1.05])
        ax2.legend(["Accuracy"], loc="center right")
        plt.title(f"{self.name} performance")
        fig.savefig(os.path.join(SAVE_PATH, "performance_" + self.name + ".png"))
        fig.clf()

        # Plot local performance
        if len(self.training_acc_loss) != 0:
            tlosses, taccs = list(zip(*[tu for arr in self.training_acc_loss for tu in arr]))
            fig, ax = plt.subplots()
            ax.plot(list(range(len(tlosses))), tlosses, color='blue', marker="o", markersize=3)
            ax.set_xlabel("Local Epochs")
            ax.set_xticklabels(self.signals[:-2], rotation=45)
            ax.set_ylabel('Loss')
            ax.legend(["Loss"], loc="center left")
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xticks((np.arange(len(self.signals[:-2])) * self.epochs) - 0.5)
            ax.set_xticklabels(self.signals[:-2])
            ax.grid()

            ax2 = ax.twinx()
            ax2.plot(list(range(len(taccs))), taccs, color='orange', marker="o", markersize=3)
            ax2.set_ylabel('Accuracy')
            for i, sig in zip(ax.get_xticks(), self.signals[:-2]):
                if sig == "Skip":
                    ax2.fill_between([i, i + self.epochs], -1, 2, color="grey", alpha=0.5)
            ax2.set_ylim([-0.05, 1.05])
            ax2.legend(["Accuracy"], loc="center right")
            plt.title(f"{self.name} local performance")
            fig.savefig(os.path.join(SAVE_PATH, "local_performance_" + self.name + ".png"))
            # plt.show()

        with open(os.path.join(SAVE_PATH, "configuration.txt"), 'a') as f:
            f.write(f"Information from {self.name}:\n\n")
            f.write(f"Signals: {self.signals}\n")
            f.write(f"Data distribution: {self.class_distribution()}\n")
            f.write(f"Accuracy: {self.accs}\n")
            f.write(f"Loss: {self.losses}\n")
            f.write(f"Training acc & loss: {self.training_acc_loss}\n")
            f.write(f"Training Acc Benign: {self.acc_benign}\n")
            f.write(f"Training Acc Poison: {self.acc_poison}\n")
            #f.write(f"Updaten Attacks in Global Rounds: {self.update_attack_rounds}\n")

            f.write(f"\n")




class adv_client_random_label(Adv_Client):
    def __init__(self, name, pr):
        super(adv_client_random_label, self).__init__(name, pr)
        self.dataloader = None

    def set_params_and_data(self, config, data_indices, model):
        super(adv_client_random_label, self).set_params_and_data(config,data_indices,model)
        self.dataloader = torch.utils.data.DataLoader(self.poison(), batch_size=config["batch_size"], shuffle=False,drop_last=True)
        for x,y in self.dataloader:
            print(x,y)
            break
        self.update()



    def poison(self):
        x_l = None
        y_l = None
        for x,y in self.dataloader:
            if x_l is None:
                x_l, y_l = x,y
            else:
                x_l = torch.cat((x_l,x))
                y_l = torch.cat((y_l,y))
        y_l = torch.squeeze(torch.randint(0,9, size = y_l.shape),dim = 0).to(device)
        self.logger.info(np.unique(y_l.cpu()))
        return TensorDataset(x_l, y_l)

class Adv_client_backdoor(Adv_Client):
    def __init__(self, name, pr):
        super(Adv_client_backdoor, self).__init__(name, pr)
        self.dataloader = None

    def set_params_and_data(self, config, data_indices, model):
        super(Adv_client_backdoor, self).set_params_and_data(config,data_indices,model)
        self.poison_data()



    def poison_data(self, BACKDOOR_TYPE = "pattern"):
        x,_ = next(iter(self.data))
        max_val = np.max(x.numpy())
        x_l = None
        y_l = None
        for x, y in self.dataloader:
            if x_l is None:
                x_l, y_l = x.numpy(), y.numpy()
            else:
                x_l = np.concatenate((x_l, x.numpy()))
                y_l = np.concatenate((y_l, y.numpy()))

        def add_modification(x):
            if BACKDOOR_TYPE == 'pattern':
                return add_pattern_bd(x, pixel_value=max_val)
            elif BACKDOOR_TYPE == 'pixel':
                return add_single_bd(x, pixel_value=max_val)
            else:
                raise ("Unknown backdoor type")

        def poison_dataset(x_clean, y_clean, percent_poison, poison_func):
            x_poison = np.copy(x_clean)
            y_poison = np.copy(y_clean)
            is_poison = np.zeros(len(y_poison))

            sources = np.arange(10)  # 0, 1, 2, 3, ...
            targets = (np.arange(10) + 1) % 10  # 1, 2, 3, 4, ...
            for i, (src, tgt) in enumerate(zip(sources, targets)):
                n_points_in_tgt = np.size(np.where(y_clean == tgt))
                num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
                src_imgs = x_clean[y_clean == src]

                n_points_in_src = np.shape(src_imgs)[0]
                indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

                imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
                backdoor_attack = PoisoningAttackBackdoor(poison_func)
                imgs_to_be_poisoned, poison_labels = backdoor_attack.poison(imgs_to_be_poisoned,
                                                                            y=np.ones(num_poison) * tgt)
                x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
                y_poison = np.append(y_poison, poison_labels, axis=0)
                is_poison = np.append(is_poison, np.ones(num_poison))

            is_poison = is_poison != 0

            return is_poison, x_poison, y_poison

        # Poison training data

        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = poison_dataset(np.transpose(x_l, (0,3,2,1)), y_l, self.poison_rate,
                                                                           add_modification)
        x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
        # Add channel axis:




        # Shuffle training data
        n_train = np.shape(y_train)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]
        is_poison_train = is_poison_train[shuffled_indices]


if __name__ == '__main__':
    s = Adv_Client("Test")

