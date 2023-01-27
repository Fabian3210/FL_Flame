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
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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

    def set_params_and_data(self, config, data_indices, model):
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

        print("hey")
        poisoned_x = []
        ys = []
        for x,y in self.dataloader:
            if np.random.rand() > self.poison_rate:
                [poisoned_x.append(t) for t in x.numpy()]
            else:
                x_adv = None
                x_adv = self.poison.generate(x.numpy(), x_adv_init=x_adv)
                [poisoned_x.append(t) for t in x_adv]

            [ys.append(t) for t in y.numpy()]
        my_dataset = TensorDataset(torch.from_numpy(np.array(poisoned_x)), torch.from_numpy(np.array(ys)))  # create your datset

        self.dataloader = DataLoader(my_dataset, batch_size=config["batch_size"], shuffle=True)

        self.logger.debug(f"Received parameters, data_indices and model from server and set them.")




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

    def set_params_and_data(self, config, data_indices, model):
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
        x,y = next(iter(self.dataloader))
        self.poison.generate(x.numpy())
        poisoned_x = []
        ys = []
        for x, y in self.dataloader:
            if np.random.rand() > self.poison_rate:
                [poisoned_x.append(t) for t in x.numpy()]
            else:
                poisoned = self.poison.apply_patch(x.numpy(),scale = 0.5)
                [poisoned_x.append(t) for t in poisoned]

            [ys.append(t) for t in y.numpy()]
        my_dataset = TensorDataset(torch.from_numpy(np.array(poisoned_x)),
                                   torch.from_numpy(np.array(ys)))  # create your datset

        self.dataloader = DataLoader(my_dataset, batch_size=config["batch_size"], shuffle=True)
        self.logger.debug(f"Received parameters, data_indices and model from server and set them.")



if __name__ == '__main__':
    s = Adv_client_ap("Test")

