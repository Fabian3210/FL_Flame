import random

import torch
from torch import nn
import time, os
from models import *
from server import Server
from flame_server import Flame_server
from client import Client
from adv_client import Adv_client_ba, Adv_client_ap
from adv_clients import *
from config import SAVE_PATH
from utils import LESS_DATA, SERVER_TEST_SIZE, SERVER_TRAIN_SIZE

def main():

    fed_config = {"C": 0.8, # percentage of clients to pick (floored)
                  "K": 50, # clients overall
                  "R": 30, # rounds of training
                  "E": 3,
                  "B": 32,
                  "ADV_bd": 0,
                  "ADV_rl": 0,
                  "ADV_mp": 0,
                  "poison_rate": 0.8,
                  "optimizer": torch.optim.Adam,
                  "criterion": nn.CrossEntropyLoss(),
                  "lr": 0.0001,
                  "data_name": "MNIST",
                  "shards_each": 2,
                  "iid": True,
                  "degree_niid": 0.9, # 0.8 + 1
                  "flame": True
                  }

    model = Net_2()

    clients = []

    for i in range(fed_config["K"]-(fed_config['ADV_bd']+fed_config["ADV_rl"]+ fed_config["ADV_mp"])):
        clients.append(Client(f"Client_{i + 1}"))
    s = 0
    for i in range(fed_config["ADV_bd"]):
        clients.append(Adv_client_backdoor(f"Adv_Client_{s}_backdoor",fed_config["poison_rate"]))
        s = s + 1
    for i in range(fed_config["ADV_rl"]):
        clients.append(Adv_client_random_label(f"Adv_Client_{s}_random_label", fed_config["poison_rate"]))
        s = s + 1
    for i in range(fed_config["ADV_mp"]):
        clients.append(Adv_client_model_poisoning(f"Adv_Client_{s}_model_poison",fed_config["poison_rate"]))
        s = s + 1

    if fed_config["flame"]:
        server = Flame_server(model, fed_config, clients)
    else:
        server = Server(model, fed_config, clients)    

    # Save configurations
    with open(os.path.join(SAVE_PATH, "configuration.txt"), 'w') as f:
        f.write(f"The following training was conducted:\n\n")
        for key, value in fed_config.items():
            f.write(f"{key}: {value}\n")
        f.write(f"model: {type(model)}\n")
        f.write(f"LESS_DATA: {LESS_DATA}\n")
        f.write(f"SERVER_TEST_SIZE: {SERVER_TEST_SIZE}\n")
        f.write(f"SERVER_TRAIN_SIZE: {SERVER_TRAIN_SIZE}\n\n\n")
    start = time.time()
    server.run()

    with open(os.path.join(SAVE_PATH, "configuration.txt"), 'a') as f:
        dur = time.time()-start
        f.write(f"Duration: {int(dur//60)} minutes {round(dur%60)} seconds\n")
    print("Finished!")

if __name__ == '__main__':
    main()
