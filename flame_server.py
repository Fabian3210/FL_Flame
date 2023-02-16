import torch
import copy, os, time
import hdbscan
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from config import SAVE_PATH, device
from utils import get_data, split_data_by_indices
from models import *
from server import Server
from matplotlib.ticker import MaxNLocator

import logging
logging.basicConfig(#filename=os.path.join(SAVE_PATH, "server_logger.txt"),
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Flame_server(Server):
    def __init__(self, model, fed_config, clients):
        super().__init__(model, fed_config, clients)
        self.past_S = []
        self.past_sigma = []
        self.tpnp = []
        self.adv_clients = ["Adv" in name for name in self.clients_names]

    def run(self):
        self.setup_data_and_model()

        self.fit()

        # Save results to file
        with open(os.path.join(SAVE_PATH, "configuration.txt"), 'a') as f:
            f.write(f"Information from Server:\n\n")
            f.write(f"Accuracy: {self.accs}\n")
            f.write(f"TPNP: {self.tpnp}\n")
            f.write(f"S: {self.past_S}\n")
            f.write(f"sigma: {self.past_sigma}\n\n")


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
        # plt.show()

        for client in self.clients:
            self.send(client, "Finish")
        self.logger.info("Exiting.")

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

        '''
        for i, client_weights in enumerate(client_models):
            cmodel = copy.deepcopy(self.model)
            cmodel.load_state_dict(client_weights)
            loss, acc = self.evaluate(eval_model=cmodel)
            self.logger.info(f"{self.clients_names[i]} values BEFORE FedAvg: loss: {loss:.3f}, accuracy: {acc:.3f}.")
        '''

        flattened, benign_client_models = self.dynamic_model_filtering(client_models)
        S, clipped_client_models = self.adaptive_clipping(flattened, benign_client_models)
        self.average_model(clipped_client_models, np.full(len(clipped_client_models), 1/len(clipped_client_models)))

        new_state_dict = self.adaptive_noising(S, self.model.state_dict())
        self.model.load_state_dict(new_state_dict)

    def fit(self):
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
        self.logger.info(f"All S: {self.past_S}")
        self.logger.info(f"All sigma: {self.past_sigma}")

    def dynamic_model_filtering(self, client_models):
        # 1. Dynamic Model Filtering
        flattened = []
        for model in client_models:
            tensor = torch.cat([torch.flatten(weight_tensor) for weight_tensor in model.values()], 0)
            tensor = tensor.to("cpu")
            flattened.append(tensor.numpy())
        cos_distances = cosine_distances(flattened, flattened)
        cos_distances = cos_distances.astype("double")
        clusterer = hdbscan.HDBSCAN(metric="precomputed",
                                    min_cluster_size=(int(self.num_clients/2)+1),
                                    min_samples=1,
                                    allow_single_cluster=True)
        clusterer.fit(X=cos_distances)
        labels = np.array(clusterer.labels_)
        sorted_list = sorted(Counter(labels).items(), key=lambda x:x[1])
        benign_cluster = sorted_list[-1][0]
        labels = labels == benign_cluster
        tpnp2 = [f"{np.sum((labels == True) & (np.array(self.adv_clients) == False))}/{self.num_clients - np.count_nonzero(self.adv_clients)}", f"{np.sum((labels == False) & (np.array(self.adv_clients) == True))}/{np.count_nonzero(self.adv_clients)}"]
        self.tpnp.append(tpnp2)
        self.logger.info(f"(1) Dynamic Model Filtering | {sorted_list}, benign cluster: {benign_cluster}, TP: {tpnp2[0]}, TN: {tpnp2[1]}")
        benign_client_models = [model for model, benign in zip(client_models, labels == np.full(self.num_clients, benign_cluster)) if benign]
        return flattened, benign_client_models

    def adaptive_clipping(self, flattened, benign_client_models):
        # 2. Adaptive Clipping
        flattened_global_model = torch.cat([torch.flatten(weight_tensor) for weight_tensor in self.model.state_dict().values()], 0)
        flattened_global_model = flattened_global_model.to("cpu").numpy()
        l2_norms = [np.sqrt(np.sum(np.square(np.subtract(flattened_global_model, model)))) for model in flattened]
        S = np.median(l2_norms)
        if S == 0:
            self.logger.info("(2) Adaptive Clipping       | Median of L2-norms equals zero, therefore, S was set to np.finfo(np.float32)!")
            S = np.finfo(np.float32).tiny
        self.past_S.append(S)
        gammas = [l2/S for l2 in l2_norms]
        clipped_client_models = []
        for model, gamma in zip(benign_client_models, gammas):
            temp_model = copy.deepcopy(model)
            for key in model.keys():
                temp_model[key] = self.model.state_dict()[key] + (model[key] - self.model.state_dict()[key]) * np.min([1, gamma])
            clipped_client_models.append(temp_model)
        self.logger.info(f"(2) Adaptive Clipping       | S: {S}")
        return S, clipped_client_models

    def adaptive_noising(self, S, state_dict):
        # 3. Adaptive Noising
        epsilon, delta = 100, 1.24 #TODO: Choose proper values
        lambd = 1/epsilon * np.sqrt(2 * np.log(1.25/delta))
        sigma = lambd * S
        self.past_sigma.append(sigma)
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to("cpu") + np.random.normal(0, sigma, state_dict[key].shape)
        self.logger.info(f"(3) Adaptive Noising        | epsilon: {epsilon}, delta: {delta}, lambda: {lambd}, sigma: {sigma}")
        return state_dict