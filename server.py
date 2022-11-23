import torch
import copy, os, time
import numpy as np
import matplotlib.pyplot as plt
from config import SAVE_PATH, device
from utils import get_data, split_data_by_indices
from models import *
from matplotlib.ticker import MaxNLocator

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Server():
    def __init__(self, model, fed_config, clients):
        # Setting all required configurations
        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]
        self.optimizer = fed_config["optimizer"]
        self.criterion = fed_config["criterion"]
        self.learning_rate = fed_config["lr"]

        self.model = model
        self.data_name = fed_config["data_name"]
        self.iid = fed_config["iid"]
        self.shards_each = fed_config["shards_each"]
        self.clients = clients
        self.clients_data_len = []
        self.clients_names = [c.name for c in self.clients]

        self.cur_round = 1

        self.losses = []
        self.accs = []

        # Setup logger
        self.logger = self.setup_logger()

    def run(self):
        self.setup_data_and_model()

        self.fit()

        # Save results to file
        with open(os.path.join(SAVE_PATH, "configuration.txt"), 'a') as f:
            f.write(f"Information from Server:\n\n")
            f.write(f"Accuracy: {self.accs}\n")
            f.write(f"Loss: {self.losses}\n")

        # Plot performance
        fig, ax = plt.subplots()
        ax.plot(list(range(len(self.losses))), self.losses, color='blue', marker="o", markersize=3)
        ax.set_xlabel("Global Rounds")
        ax.set_ylabel('Loss')
        ax.legend(["Loss"], loc="center left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax2 = ax.twinx()
        ax2.plot(list(range(len(self.accs))), self.accs, color='orange', marker="o", markersize=3)
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([-0.05, 1.05])
        ax2.legend(["Accuracy"], loc="center right")
        ax.grid()

        plt.title(f"Server Performance")
        fig.savefig(os.path.join(SAVE_PATH, "performance_server.png"))

        for client in self.clients:
            self.send(client, "Finish")
        self.logger.info("Exiting.")


    def setup_data_and_model(self):
        """
        Setup server and clients. This method gets the test data, splits the train data indices and sends everything
        important to the clients (data name, indices, config, model).
        """
        self.logger.debug("Start data & model setup.")

        self.data = get_data(self.data_name, train=True)
        self.test_data, _ = get_data(self.data_name, train=False)

        self.test_dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

        splits = split_data_by_indices(self.data, self.num_clients, iid=self.iid, shards_each=self.shards_each)

        client_config = {"data_name": self.data_name,
                         "epochs": self.local_epochs,
                         "batch_size": self.batch_size,
                         "optimizer": self.optimizer,
                         "learning_rate": self.learning_rate,
                         "criterion": self.criterion,
                         "data_name": self.data_name}
        for i, client in enumerate(self.clients):
            client.set_params_and_data(client_config, splits[i], copy.deepcopy(self.model))
            self.clients_data_len.append(len(splits[i]))

        self.logger.info("Data & model setup finished.")

    def average_model(self, client_models, coefficients):
        """
        Average the global model with the clients models. These will be weighted according to the coefficients.

        :param client_models: State dicts of the clients models in a list
        :param coefficients: Coefficients in a list
        """
      
        global_dict = self.model.state_dict()
        averaged_dict = OrderedDict()
        for layer in global_dict.keys():
            averaged_dict[layer] = torch.zeros(global_dict[layer].shape, dtype=torch.float32)
            for client_dict, coef in zip(client_models, coefficients):
                client_dict[layer] = client_dict[layer].to("cpu")
                averaged_dict[layer] += coef * client_dict[layer]

        self.model.load_state_dict(averaged_dict)

    def train(self):
        """
        One round of training. Consisting of choosing the fraction of clients involved in this round, updating the
        clients models, collecting the updates and averaging the global model.
        """
        m = np.maximum(int(self.fraction * len(self.clients)), 1)
        indices = np.random.choice(self.num_clients, m, replace=False)

        signals = ["Skip"] * self.num_clients
        list(map(signals.__setitem__, indices, ["Update"] * len(indices)))

        client_models = [self.send(client, signal) for client, signal in zip(self.clients, signals)]
        coefficients = [size/sum(self.clients_data_len) for size in self.clients_data_len]
        self.average_model(client_models,coefficients)

    def fit(self):
        """
        Starts the training for the specified number of rounds.
        """
        self.logger.debug("Start training...")

        for r in range(1, self.num_rounds + 1):
            start = time.time()
            self.cur_round += 1
            self.logger.debug(f"Round {r}/{self.num_rounds}...")
            self.train()
            loss, acc = self.evaluate()
            self.losses.append(loss)
            self.accs.append(acc)
            dur = time.time() - start
            self.logger.info(f"Round {r}/{self.num_rounds} completed ({int(dur//60)} min {int(dur%60)} sec): loss: {loss:.3f}, accuracy: {acc:.3f}.")

        for client in self.clients:
            self.send(client, "Finish")

        self.logger.info("Finished training!")

    def evaluate(self, eval_model = None):
        """
        Evaluate the current global model with the specified test data.

        :return: loss: loss of the model given its data
        :return: acc : accuracy of the model given its data
        """
        if eval_model is None:
            eval_model = self.model
        eval_model = eval_model.to(device)
        eval_model.eval()
        loss = 0
        acc = 0
        with torch.no_grad():
            for x, y in self.test_dataloader:
                x = x.to(device)
                y = y.to(device)
                outputs = eval_model(x)
                loss += self.criterion(outputs, y).item()
                preds = outputs.argmax(dim=1, keepdim=True)
                acc += (preds == y.view_as(preds)).sum().item()
        loss = loss / len(self.test_dataloader)
        acc = acc / len(self.test_data)

        return loss, acc

    def setup_logger(self, name="Server"):
        logger = logging.getLogger(name)
        #logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        return logger

    def send(self,client, signal):
        self.logger.debug(f"Server --{signal}--> {client.name}")
        if signal == "Update":
            self.logger.debug(f"Server --Model--> {client.name}")
            return client.receive(signal, copy.deepcopy(self.model.state_dict()))
        elif signal == "Skip":
            return client.receive(signal, 0)
        elif signal == "Finish":
            self.logger.debug(f"Server --Model--> {client.name}")
            client.receive(signal, copy.deepcopy(self.model.state_dict()))
            return
        #self.logger.debug(f"Server <--Model-- {name}")
