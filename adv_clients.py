import copy
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

    def evaluate(self, eval_model=None):
        self.adv_metrics()
        return super().evaluate()


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
        self.acc_benign.append(acc)
        self.acc_poison.append(adv_acc)
        return acc, adv_acc

    def finish_function(self):
        benign_acc, adv_acc = self.acc_benign[-1], self.acc_poison[-1]
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




class Adv_client_random_label(Adv_Client):
    def __init__(self, name, pr):
        super(Adv_client_random_label, self).__init__(name, pr)
        self.dataloader = None

    def set_params_and_data(self, config, data_indices, model):
        super(Adv_client_random_label, self).set_params_and_data(config,data_indices,model)
        self.dataloader = torch.utils.data.DataLoader(self.poison_data(), batch_size=config["batch_size"], shuffle=True)

    def poison_data(self):
        x_l = None
        y_l = None
        y_l_true = None
        self.poison_ind = np.zeros(len(self.dataloader))
        for i, (x, y) in enumerate(self.dataloader):

            if x_l is None:
                r = np.random.random()
                if r < self.poison_rate:

                    x_l, y_l, y_l_true = x.numpy(), np.random.randint(low=0, high=9, size=y.shape), y.numpy()
                    self.poison_ind[i] = 1
                else:
                    x_l, y_l, y_l_true = x.numpy(), y.numpy(), y.numpy()
            else:
                x_l = np.concatenate((x_l, x.numpy()))
                y_l_true = np.concatenate((y_l_true, y.numpy()))
                r = np.random.random()
                if r < self.poison_rate:
                    y_l = np.concatenate((y_l, np.random.randint(low= 0,high= 9, size = y.shape)))
                    self.poison_ind[i] = 1
                else:
                    y_l = np.concatenate((y_l, y.numpy()))
        y_l = torch.Tensor(y_l).long()
        x_l = torch.Tensor(x_l)
        self.logger.info(f"Amount of true labels: {(sum(y_l == torch.Tensor(y_l_true).long())/200):.3f}")
        return TensorDataset(x_l, y_l)

class Adv_client_backdoor(Adv_Client):
    def __init__(self, name, pr):
        super(Adv_client_backdoor, self).__init__(name, pr)
        self.dataloader = None

    def set_params_and_data(self, config, data_indices, model):
        super(Adv_client_backdoor, self).set_params_and_data(config,data_indices,model)
        self.dataloader = torch.utils.data.DataLoader(self.poison_data(),batch_size=config["batch_size"], shuffle=True)



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
            x_poison = None
            y_poison = None
            is_poison = None

            sources = np.arange(10)  # 0, 1, 2, 3, ...
            targets = np.arange(1,11) % 10   # 1, 2, 3, 4, ...

            for i, (src, tgt) in enumerate(zip(sources, targets)):

                src_imgs = x_clean[y_clean == src]
                num_poison = round(len(src_imgs) * percent_poison)
                if len(src_imgs) == 0:
                    continue


                n_points_in_src = np.shape(src_imgs)[0]
                indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

                imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
                if self.save_example:
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_x.png"), imgs_to_be_poisoned[0].tolist(), cmap='gray')

                backdoor_attack = PoisoningAttackBackdoor(poison_func)
                imgs_to_be_poisoned, poison_labels = backdoor_attack.poison(imgs_to_be_poisoned,
                                                                            y=np.ones(num_poison) * tgt)
                if self.save_example:
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_x_poisoned.png"), imgs_to_be_poisoned[0].tolist(), cmap='gray')
                    self.save_example = False
                if x_poison is None:
                    x_poison = imgs_to_be_poisoned
                    y_poison = poison_labels
                    is_poison = np.ones(indices_to_be_poisoned.shape)
                else:
                    x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
                    y_poison = np.append(y_poison, poison_labels, axis=0)
                    is_poison = np.append(is_poison, np.ones(num_poison))

            is_poison = is_poison != 0

            return is_poison, x_poison, y_poison

        # Poison training data

        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = poison_dataset(np.transpose(x_l, (0,2,3,1)), y_l, self.poison_rate,
                                                                           add_modification)
        x_train, y_train = preprocess(x_poisoned_raw, y_poisoned_raw)
        x_train = torch.Tensor(np.transpose(x_train, (0,3,1,2)))
        y_train = torch.Tensor(y_train.argmax(axis = 1)).long()

        x_train_b = torch.Tensor(x_l)
        y_train_b = torch.Tensor(y_l).long()

        self.dataloader_benign = torch.utils.data.DataLoader(TensorDataset(x_train_b,y_train_b),batch_size=self.batch, shuffle=True)
        return TensorDataset(x_train,y_train)

    def adv_metrics(self):
        self.model.eval()
        eval_benign = []
        eval_adversial = []
        for x, y in self.dataloader:
            x = x.to(device)
            y = y.to(device)
            self.model.to(device)

            pred: object = self.model(x).max(dim=1).indices
            eval_adversial.append((torch.sum(pred == y) / len(y)).cpu())
        for x, y in self.dataloader_benign:
            x = x.to(device)
            y = y.to(device)
            self.model.to(device)

            pred: object = self.model(x).max(dim=1).indices
            eval_benign.append((torch.sum(pred == y) / len(y)).cpu())

        acc = np.mean(eval_benign)
        adv_acc = np.mean(eval_adversial)
        self.acc_benign.append(acc)
        self.acc_poison.append(adv_acc)
        return acc, adv_acc

class Adv_client_model_poisoning(Adv_Client):
    def __init__(self, name, pr):
        super(Adv_client_model_poisoning, self).__init__(name, pr)
        self.dataloader = None
        self.benign_model = None

    def set_params_and_data(self, config, data_indices, model):
        super(Adv_client_model_poisoning, self).set_params_and_data(config, data_indices, model)
        self.benign_model = copy.deepcopy(self.model)

    def update(self):
        super(Adv_client_model_poisoning, self).update()
        self.poison_data()


    def poison_data(self):
        self.benign_model = copy.deepcopy(self.model)
        self.benign_model.load_state_dict(self.model.state_dict())
        state_dict = {}
        for key, value in self.model.state_dict().items():
            len = 1
            for x in value.shape: len *= x
            if len == 1:
                state_dict[key] = value
                continue
            mul = np.concatenate([np.random.uniform(-self.poison_rate, self.poison_rate, int(np.floor(len/2))) + 1,
                                  np.random.uniform(-self.poison_rate, self.poison_rate, int(np.ceil(len/2))) + 1])
            np.random.shuffle(mul)
            mul = np.reshape(mul, list(value.shape))

            mul = torch.Tensor(mul).to(device)
            state_dict[key] = torch.multiply(copy.deepcopy(value), mul)
        self.model.load_state_dict(state_dict)

    def adv_metrics(self):
        self.model.eval()
        self.benign_model.eval()
        eval_benign = []
        eval_adversial = []
        for x, y in self.dataloader:
            x = x.to(device)
            y = y.to(device)
            self.model.to(device)

            pred: object = self.model(x).max(dim=1).indices
            eval_adversial.append((torch.sum(pred == y) / len(y)).cpu())
            pred: object = self.benign_model(x).max(dim=1).indices
            eval_benign.append((torch.sum(pred == y) / len(y)).cpu())

        acc = np.mean(eval_benign)
        adv_acc = np.mean(eval_adversial)
        self.acc_benign.append(acc)
        self.acc_poison.append(adv_acc)
        return acc, adv_acc



if __name__ == '__main__':
    s = Adv_client("Test")

