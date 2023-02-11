import random

import torch
from torch import nn
import time, os
from models import *
from server import Server
from flame_server import Flame_server
from client import Client
from adv_client import Adv_client_ba, Adv_client_ap, Adv_client_backdoor
from adv_clients import adv_client_random_label
from config import SAVE_PATH
from utils import LESS_DATA, SERVER_TEST_SIZE, SERVER_TRAIN_SIZE

def main():

    fed_config = {"C": 1, # percentage of clients to pick (floored)
                  "K": 5, # clients overall
                  "R": 10, # rounds of training
                  "E": 3,
                  "B": 64,
                  "A": 5,
                  "A_random": False,
                  "ADV_ap": 0,
                  "ADV_ba": 0,
                  "ADV_bd": 1,
                  "poison_rate": 0.75, #TODO: Poison Rates in Literatures
                  "optimizer": torch.optim.Adam,
                  "criterion": nn.CrossEntropyLoss(),
                  "lr": 0.001,
                  "data_name": "FashionMNIST",
                  "shards_each": 2,
                  "iid": True,
                  "degree_niid": 0.8, # 0.8 + 1
                  "flame": True
                  }

    model = FashionCNN()

    clients = []
    if fed_config["A_random"]:
        for i in range(fed_config["K"]-fed_config["A"]):
            clients.append(Client(f"Client_{i + 1}"))
        print(f"Created {fed_config['K']-fed_config['A']} normal clients.")

        for i in range(fed_config["A"]):
            r = random.random()
            if r < 0.5:
                clients.append(Adv_client_ap(f"Adv_Client_{i}_ap", fed_config["poison_rate"]))
            if r > 0.5:
                clients.append(Adv_client_backdoor(f"Adv_Client_{i}_bd", fed_config["poison_rate"]))
    else:
        for i in range(fed_config["K"]-(fed_config["ADV_ap"]+fed_config["ADV_ba"] +fed_config['ADV_bd'])):
            clients.append(Client(f"Client_{i + 1}"))
        print(f"Created {fed_config['K'] - (fed_config['ADV_ap'] + fed_config['ADV_ba']+ fed_config['ADV_bd'])} normal clients.")
        s = 0
        for i in range(fed_config["ADV_ap"]):
            clients.append(Adv_client_ap(f"Adv_Client_{s}_ap",fed_config["poison_rate"]))
            s = s + 1

        for i in range(fed_config["ADV_ba"]):
            clients.append(Adv_client_ba(f"Adv_Client_{s}_ba", fed_config["poison_rate"]))
            s = s + 1
        for i in range(fed_config["ADV_bd"]):
            clients.append(adv_client_random_label(f"Adv_Client_{s}_random_label",fed_config["poison_rate"]))
            s = s + 1
    if (fed_config["ADV_ap"]+fed_config["ADV_ba"]+fed_config["ADV_bd"]) != 0:
        print("Created the following Adversial Clients:")
        for i in clients:
            if "Adv" in i.name: print(i.name)
    
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
