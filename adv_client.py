import os
import time
import torch
from torch.utils.data import TensorDataset, DataLoader

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from art.estimators.classification import PyTorchClassifier
from config import SAVE_PATH, device
from utils import get_data_by_indices
from client import Client
import logging
from random import random
from matplotlib.ticker import MaxNLocator
from art.attacks.evasion import BoundaryAttack, AdversarialPatchPyTorch

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib').setLevel(logging.INFO)

class Adv_client_ba(Client): #TODO: Implement adersary
    def __init__(self, name, pr = 0.3):
        super(Client).__init__()
        self.name = name
        self.logger = self.setup_logger(self.name)
        self.accs = []
        self.losses = []
        self.training_acc_loss = []
        self.signals = []
        self.batch = None
        self.model = None

        self.poison = None
        self.poison_rate = pr
        self.save_example = True
        self.retrain_accuracy = 0.9



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
            targeted=False,
            max_iter = 250,
            delta= 0.01,
            epsilon= 0.0001)
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

        print(f"Accuracy Benign: {np.mean(eval_benign)} for {np.count_nonzero(self.poison_ind == False)} / {len(self.poison_ind)}")
        print(f"Accuracy Adversial: {np.mean(eval_adversial)} for {np.count_nonzero(self.poison_ind == True)} / {len(self.poison_ind)}")
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
        for x,y in self.dataloader:
            IND = IND + 1
            if np.random.rand() > self.poison_rate:
                [poisoned_x.append(t) for t in x.numpy()]
            else:
                x_adv = None
                x_adv = self.poison.generate(x.numpy(), x_adv_init=x_adv)
                [poisoned_x.append(t) for t in x_adv]
                self.poison_ind[IND] = 1
                if self.save_example:
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_x.png"), x[0][0])
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_x_poisoned.png"), x_adv[0][0])
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_poison.png"), (x_adv[0][0]- x[0][0].numpy()))
                    self.save_example = False

            [ys.append(t) for t in y.numpy()]

        my_dataset = TensorDataset(torch.from_numpy(np.array(poisoned_x)),
                                   torch.from_numpy(np.array(ys)))  # create your datset

        self.dataloader = DataLoader(my_dataset, batch_size=self.config["batch_size"], shuffle=True)



class Adv_client_ap(Client): #TODO: Implement adersary
    def __init__(self, name, pr = 0.3):
        super(Client).__init__()
        self.name = name
        self.logger = self.setup_logger(self.name)
        self.accs = []
        self.losses = []
        self.training_acc_loss = []
        self.signals = []
        self.batch = None
        self.model = None

        self.poison = None
        self.poison_rate = pr
        self.save_example = True
        self.retrain_accuracy = 0.85

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
            clip_values=(-1, 1),
            loss=self.criterion,
            optimizer=self.optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,
        )
        batch_size = self.batch
        scale_min = 0.4
        scale_max = 1.0
        rotation_max = 22.5
        learning_rate = 5000.
        max_iter = 250

        self.poison = AdversarialPatchPyTorch(estimator=classifier_py, rotation_max=rotation_max, scale_min=scale_min,
                                     scale_max=scale_max,
                                     learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size,
                                     patch_shape=(1, 7, 7))
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
        print(f"Accuracy Benign: {np.mean(eval_benign)} for {np.count_nonzero(self.poison_ind == False)} / {len(self.poison_ind)}")
        print(f"Accuracy Adversial: {np.mean(eval_adversial)} for {np.count_nonzero(self.poison_ind == True)} / {len(self.poison_ind)}")
        if np.mean(eval_adversial) > self.retrain_accuracy:
            self.logger.info("Retraining: ", self.name)
            print("Retraining", self.name)
            self.update_attack()
        super().update()

    def update_attack(self):

        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.config["batch_size"], shuffle=True)
        x, y = next(iter(self.dataloader))
        self.model.to("cpu")
        self.poison.generate(x.numpy())
        self.poison_ind = np.zeros(int(np.ceil(len(self.dataloader.dataset) / self.batch)))
        poisoned_x = []
        ys = []
        IND = -1
        for x, y in self.dataloader:
            IND = IND + 1
            if np.random.rand() > self.poison_rate:
                [poisoned_x.append(t) for t in x.numpy()]
            else:
                poisoned = self.poison.apply_patch(x.numpy(), scale=0.5)
                self.poison_ind[IND] = 1
                [poisoned_x.append(t) for t in poisoned]
                if self.save_example:
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_x.png"), x[0][0])
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_x_poisoned.png"), poisoned[0][0])
                    plt.imsave(os.path.join(SAVE_PATH, f"{self.name}_poison.png"), (poisoned[0][0] - x[0][0].numpy()))
                    self.save_example = False

            [ys.append(t) for t in y.numpy()]

        my_dataset = TensorDataset(torch.from_numpy(np.array(poisoned_x)),
                                   torch.from_numpy(np.array(ys)))  # create your datset

        self.dataloader = DataLoader(my_dataset, batch_size=self.config["batch_size"], shuffle=True)

if __name__ == '__main__':
    s = Adv_client_ap("Test")

