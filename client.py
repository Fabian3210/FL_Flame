import os
import time
import torch
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from config import SAVE_PATH, device
from utils import get_data_by_indices
import logging
from random import random
from matplotlib.ticker import MaxNLocator

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Client():
    def __init__(self, name):
        self.name = name
        self.logger = self.setup_logger(self.name)
        self.accs = []
        self.losses = []
        self.training_acc_loss = []
        self.signals = []
        self.batch = 0

    def set_params_and_data(self, config, data_indices, model):
        self.epochs = config["epochs"]
        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]
        self.criterion = config["criterion"]
        self.batch = config["batch_size"]

        self.data = get_data_by_indices(config["data_name"], True, data_indices)
        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch, shuffle=True)
        self.logger.info(f"Data distribution: {str(self.class_distribution())}")

        self.model = model

        self.logger.debug(f"Received parameters, data_indices and model from server and set them.")

    def update(self):
        """
        Does one round of training for the specified number of epochs.
        """
        self.logger.debug(f"Start training...")

        self.model = self.model.to(device)
        self.model.train()

        temp_performance = []
        optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            start = time.time()
            self.logger.debug(f"Epoch {epoch+1}/{self.epochs}...")
            for x, y in self.dataloader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()

            loss, acc = self.evaluate()
            temp_performance.append([loss, acc])
            self.logger.info(f"Epoch {epoch+1}/{self.epochs} completed ({int(time.time()-start)} sec): loss: {loss:.3f}, acc: {acc:.3f}.")

        self.training_acc_loss.append(temp_performance)
        self.logger.info("...finished training!")

    def evaluate(self, eval_model=None):
        """
        Evaluate the current client's model with its data

        :return: loss: loss of the model given its data
                 acc: accuracy of the model given its data
        """
        if eval_model is None:
            eval_model = self.model

        eval_model = eval_model.to(device)
        eval_model.eval()

        loss = 0
        acc = 0

        with torch.no_grad():
            for x, y in self.dataloader:
                x = x.to(device)
                y = y.to(device)
                outputs = eval_model(x)
                loss += self.criterion(outputs, y).item()
                preds = outputs.argmax(dim=1, keepdim=True)
                acc += (preds == y.view_as(preds)).sum().item()

        loss = loss / len(self.dataloader)
        acc = acc / len(self.data)

        return loss, acc

    def class_distribution(self):
        '''
        Calculates the number of instances per class and retunrs it.

        :return: Class distribution of the local dataset
        '''
        np_targets = np.array(self.data.dataset.targets)
        return sorted(Counter(np_targets[self.data.indices]).items())

    def setup_logger(self, name):
        logger = logging.getLogger(name)
        # logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        return logger

    def receive(self, signal, data):
        self.signals.append(signal)
        self.logger.debug(f"{self.name} <--{signal}-- Server")
        if signal == "Update": # data = model weights
            self.model.load_state_dict(data)
            self.logger.debug(f"{self.name} <--Model-- Server")
            self.update()

            loss, acc = self.evaluate()
            self.accs.append(acc)
            self.losses.append(loss)

            self.logger.debug(f"{self.name} --Model--> Server")
            return self.model.state_dict()
        elif signal == "Skip": # data will be ignored
            self.logger.debug(f"{self.name} --Model--> Server")
            if len(self.training_acc_loss) == 0:
                self.training_acc_loss.append([[np.nan, np.nan]] * self.epochs)
            else:
                self.training_acc_loss.append([self.training_acc_loss[-1][-1]] * self.epochs)
            loss, acc = self.evaluate()
            self.accs.append(acc)
            self.losses.append(loss)

            return self.model.state_dict()
        elif signal == "Finish": # data = model weights
            self.model.load_state_dict(data)
            self.logger.debug(f"{self.name} <--Complete model-- Server")
            loss, acc = self.evaluate()
            self.accs.append(acc)
            self.losses.append(loss)
            self.logger.info(f"Metrics using final model: loss: {loss:.3f}, acc: {acc:.3f}.")
            self.finish_function()
            self.logger.info("Exiting!")
            self.plots()
            return

    def finish_function(self):
        pass

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
            plt.show()

        with open(os.path.join(SAVE_PATH, "configuration.txt"), 'a') as f:
            f.write(f"Information from {self.name}:\n\n")
            f.write(f"Signals: {self.signals}\n")
            f.write(f"Data distribution: {self.class_distribution()}\n")
            f.write(f"Accuracy: {self.accs}\n")
            f.write(f"Loss: {self.losses}\n")
            f.write(f"Training acc & loss: {self.training_acc_loss}\n")
            f.write(f"\n")
