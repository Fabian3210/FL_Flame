import os
import time
import torch
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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class Adv_client_ba(Client):
    def __init__(self, name, pr = 0.3):
        super(Client).__init__()
        self.name = name
        self.logger = self.setup_logger(self.name)
        self.logger.setLevel(logging.DEBUG) # TODO: Remove
        self.accs = []
        self.losses = []
        self.training_acc_loss = []
        self.signals = []
        self.update_attack_rounds = []
        self.batch = None
        self.model = None




        self.poison = None
        self.poison_rate = pr
        self.save_example = True
        self.retrain_accuracy = 1
        self.acc_benign = []
        self.acc_poison = []

    def setup_logger(self, name):
        logger = logging.getLogger(name)
        # logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        return logger



    def set_params_and_data(self, config, data_indices, model):
        self.config = config
        self.epochs = config["epochs"]
        self.batch = config["batch_size"]
        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]
        self.criterion = config["criterion"]

        self.data = get_data_by_indices(config["data_name"], True, data_indices)
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=config["batch_size"], shuffle=True)

        self.logger.info(f"Data distribution: {str(self.class_distribution())}")
        self.model = model
        self.model = self.model.to(device)
        classifier_py = PyTorchClassifier(
            model=self.model,
            clip_values=(-1, 1),
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,
        )

        self.poison = BoundaryAttack(
            classifier_py,
            targeted=True,
            max_iter = 500,
            delta= 0.01,
            epsilon= 0.0001, verbose=False)
        self.update_attack()
        self.logger.debug(f"Received parameters, data_indices and model from server and set them.")


    def update(self):
        #check for accuracy of missclassified data
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

        self.logger.info(f"Accuracy Benign: {np.mean(eval_benign)} for {np.count_nonzero(self.poison_ind == False)} / {len(self.poison_ind)}")
        self.logger.info(f"Accuracy Adversial: {np.mean(eval_adversial)} for {np.count_nonzero(self.poison_ind == True)} / {len(self.poison_ind)}")
        if np.mean(eval_adversial) > self.retrain_accuracy:
            self.update_attack()

        super().update()

    def update_attack(self):

        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.config["batch_size"], shuffle=True)
        self.model.to("cpu")
        self.poison_ind = np.zeros(int(np.ceil(len(self.dataloader.dataset) / self.batch)))
        poisoned_x = []
        ys = []
        IND = -1
        t = time.time()
        for x,y in self.dataloader:
            IND = IND + 1
            if np.random.rand() > self.poison_rate:
                [poisoned_x.append(t) for t in x.numpy()]
            else:
                x_adv = None
                s = np.copy(y)
                np.random.shuffle(s)
                x_adv = self.poison.generate(x=x.numpy(), y = s, x_adv_init=x_adv)
                [poisoned_x.append(t) for t in x_adv]
                self.poison_ind[IND] = 1
                if self.save_example:
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_x.png"), x[0][0], cmap='gray')
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_x_poisoned.png"), x_adv[0][0], cmap='gray')
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_poison.png"), (x_adv[0][0]- x[0][0].numpy()), cmap='gray')
                    self.save_example = False

            [ys.append(t) for t in y.numpy()]
        self.logger.info("Poison took ", int(time.time()-t), " seconds in ", self.name)

        my_dataset = TensorDataset(torch.from_numpy(np.array(poisoned_x)),
                                   torch.from_numpy(np.array(ys)))  # create your datset

        self.dataloader = DataLoader(my_dataset, batch_size=self.config["batch_size"], shuffle=True)
        self.update_attack_rounds.append(cur_round)

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
            #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xticks((np.arange(len(self.signals[:-2]))*self.epochs)-0.5)
            ax.set_xticklabels(self.signals[:-2])
            ax.grid()

            ax2 = ax.twinx()
            ax2.plot(list(range(len(taccs))), taccs, color='orange', marker="o", markersize=3)
            ax2.set_ylabel('Accuracy')
            for i, sig in zip(ax.get_xticks(), self.signals[:-2]):
                if sig == "Skip":
                    ax2.fill_between([i, i+self.epochs], -1, 2, color="grey", alpha=0.5)
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
            f.write(f"Updaten Attacks in Global Rounds: {self.update_attack_rounds}\n")

            f.write(f"\n")











class Adv_client_ap(Client):
    def __init__(self, name, pr = 0.3):
        super(Client).__init__()
        self.name = name
        self.logger = self.setup_logger(self.name)
        self.logger.setLevel(logging.INFO) # TODO: Remove
        self.accs = []
        self.losses = []
        self.training_acc_loss = []
        self.signals = []
        self.batch = None
        self.model = None
        self.update_attack_rounds = []


        self.poison = None
        self.poison_rate = pr
        self.save_example = True
        self.retrain_accuracy = 0
        self.acc_benign = []
        self.acc_poison = []

    def set_params_and_data(self, config, data_indices, model):
        self.config = config
        self.epochs = config["epochs"]
        self.batch = config["batch_size"]
        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]
        self.criterion = config["criterion"]

        self.data = get_data_by_indices(config["data_name"], True, data_indices)
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=config["batch_size"], shuffle=True)
        self.logger.info(f"Data distribution: {str(self.class_distribution())}")
        self.model = model
        classifier_py = PyTorchClassifier(
            model=self.model,
            clip_values=(0, 1),
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=self.data.dataset.data[0].shape[::-1],
            nb_classes=10,
        )
        batch_size = self.batch
        scale_min = 0.4
        scale_max = 1.0
        rotation_max = 22.5
        learning_rate = 5000.
        max_iter = 250

        self.poison = AdversarialPatchPyTorch(estimator=classifier_py, targeted=True,rotation_max=rotation_max, scale_min=scale_min,
                                     scale_max=scale_max,
                                     learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size,
                                     patch_shape=self.data.dataset.data[0].shape[::-1], verbose=False)
        self.update_attack()
        self.logger.debug(f"Received parameters, data_indices and model from server and set them.")

    def update(self):
        #check for accuracy of missclassified data
        benign_acc, adv_acc = self.adv_metrics()
        self.logger.debug(f"Accuracy Benign: {benign_acc} for {np.count_nonzero(self.poison_ind == False)} / {len(self.poison_ind)}")
        self.logger.debug(f"Accuracy Adversial: {adv_acc} for {np.count_nonzero(self.poison_ind == True)} / {len(self.poison_ind)}")
        if np.mean(adv_acc) > self.retrain_accuracy:
            self.logger.info("Retraining: ", self.name)
        self.logger.debug("Retraining the attack")
        self.update_attack()
        super().update()
        benign_acc, adv_acc = self.adv_metrics()
        self.acc_benign.append(benign_acc)
        self.acc_poison.append(adv_acc)

    def update_attack(self):

        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.config["batch_size"], shuffle=True)
        x, y = next(iter(self.dataloader))

        v, c = np.unique(y, return_counts=True)
        ind = np.argmax(c)

        self.model.to("cpu")
        self.poison.generate(x[y==ind].numpy(), y[y==ind].numpy())
        self.poison_ind = np.zeros(int(np.ceil(len(self.dataloader.dataset) / self.batch)))
        poisoned_x = []
        ys = []
        IND = -1
        t = time.time()
        for x, y in self.dataloader:
            IND = IND + 1
            if np.random.rand() > self.poison_rate:
                [poisoned_x.append(t) for t in x.numpy()]
            else:

                poisoned = self.poison.apply_patch(x.numpy(), scale=0.8)
                self.poison_ind[IND] = 1
                [poisoned_x.append(t) for t in poisoned]
                if self.save_example:
                    img = ((x[0].numpy() * 0.5) + 0.5).transpose(1, 2, 0)
                    p_img = ((poisoned[0] * 0.5) + 0.5).transpose(1, 2, 0)
                    diff = ((p_img - img) * 0.5) + 0.5
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_x.png"), img)
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_x_poisoned.png"), p_img)
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_poison.png"), diff)
                    self.save_example = False

            [ys.append(t) for t in y.numpy()]
        self.logger.info(f"Poison took {int(time.time() - t)} seconds in {self.name}")
        my_dataset = TensorDataset(torch.from_numpy(np.array(poisoned_x)),
                                   torch.from_numpy(np.array(ys)))  # create your datset

        self.dataloader = DataLoader(my_dataset, batch_size=self.config["batch_size"], shuffle=True)
        self.update_attack_rounds.append(cur_round)

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
            f.write(f"Updaten Attacks in Global Rounds: {self.update_attack_rounds}\n")

            f.write(f"\n")

class Adv_client_backdoor(Client):
    def __init__(self, name, pr=0.3):
        super(Client).__init__()
        self.name = name
        self.logger = self.setup_logger(self.name)
        self.accs = []
        self.losses = []
        self.training_acc_loss = []
        self.signals = []
        self.batch = None
        self.model = None
        self.update_attack_rounds = []

        self.poison = None
        self.poison_rate = pr
        self.save_example = True
        self.retrain_accuracy = 0
        self.acc_benign = []
        self.acc_poison = []
    def set_params_and_data(self, config, data_indices, model):
        self.config = config
        self.epochs = config["epochs"]
        self.batch = config["batch_size"]
        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]
        self.criterion = config["criterion"]

        self.data = get_data_by_indices(config["data_name"], True, data_indices)

        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=config["batch_size"], shuffle=True)


        #(is_poison_train, x_poisoned_raw, y_poisoned_raw, max_val) = self.generate_backdoor(self.data.dataset.data[self.data.indices].numpy(), self.data.dataset.targets[self.data.indices].numpy(), self.poison_rate)
        y = self.data.dataset.targets[self.data.indices]
        y[:int(len(y) * 0.8)] = 0
        x = self.data.dataset.data[self.data.indices]
        data = TensorDataset(x,y)
        self.dataloader = torch.utils.data.DataLoader(data, batch_size=config["batch_size"], shuffle=True)
        self.logger.info(f"Data distribution: {str(self.class_distribution())}")
        self.model = model
        self.logger.debug(f"Received parameters, data_indices and model from server and set them.")

    def generate_backdoor(self,
            x_clean, y_clean, percent_poison, backdoor_type="pattern", sources=np.arange(10),
            targets=(np.arange(10) + 1) % 10):

        """
        Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
        class. Default parameters poison each digit so that it gets classified to the next digit.
        :param x_clean: Original raw data
        :type x_clean: `np.ndarray`
        :param y_clean: Original labels
        :type y_clean:`np.ndarray`
        :param percent_poison: After poisoning, the target class should contain this percentage of poison
        :type percent_poison: `float`
        :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
        :type backdoor_type: `str`
        :param sources: Array that holds the source classes for each backdoor. Poison is
        generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
        Poisonous images from sources[i] will be labeled as targets[i].
        :type sources: `np.ndarray`
        :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                        labeled as targets[i].
        :type targets: `np.ndarray`
        :return: Returns is_poison, which is a boolean array indicating which points are poisonous, x_poison, which
        contains all of the data both legitimate and poisoned, and y_poison, which contains all of the labels
        both legitimate and poisoned.
        :rtype: `tuple`
        """
        from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd

        max_val = np.max(x_clean)

        x_poison = np.copy(x_clean)
        y_poison = np.copy(y_clean)
        is_poison = np.zeros(np.shape(y_poison))

        for i, (src, tgt) in enumerate(zip(sources, targets)):
            n_points_in_tgt = np.size(np.where(y_clean == tgt))
            num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
            src_imgs = x_clean[y_clean == src]

            n_points_in_src = np.shape(src_imgs)[0]
            indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

            imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
            if backdoor_type == "pattern":
                imgs_to_be_poisoned = add_pattern_bd(x=imgs_to_be_poisoned, pixel_value=max_val)
            elif backdoor_type == "pixel":
                imgs_to_be_poisoned = add_single_bd(imgs_to_be_poisoned, pixel_value=max_val)
            x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
            y_poison = np.append(y_poison, np.ones(num_poison) * tgt, axis=0)
            is_poison = np.append(is_poison, np.ones(num_poison))

        is_poison = is_poison != 0

        return is_poison, x_poison, y_poison, max_val




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
            f.write(f"Updaten Attacks in Global Rounds: {self.update_attack_rounds}\n")

            f.write(f"\n")



if __name__ == '__main__':
    s = Adv_client_ap("Test")

