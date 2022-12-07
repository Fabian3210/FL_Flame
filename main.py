import torch
from torch import nn
import time, os
from models import *
from server import Server
from flame_server import Flame_server
from client import Client
from adv_client import Adv_client
from config import SAVE_PATH
from utils import LESS_DATA, SERVER_TEST_SIZE, SERVER_TRAIN_SIZE

def main():

    fed_config = {"C": 0.5, # percentage of clients to pick (floored)
                  "K": 10, # clients overall
                  "R": 5, # rounds of training
                  "E": 3,
                  "B": 64,
                  "optimizer": torch.optim.Adam,
                  "criterion": nn.CrossEntropyLoss(),
                  "lr": 0.01,
                  "data_name": "MNIST",
                  "shards_each": 2,
                  "iid": True
                  }

    if fed_config["data_name"] == "MNIST":
        model = Net_2()
    elif fed_config["data_name"] == "FashionMNIST":
        model = Net_2()
    elif fed_config["data_name"] == "CIFAR100":
        model = Net_4(100)
    elif fed_config["data_name"] == "CIFAR10":
        model = Net_4(10)
    else:
        raise AssertionError("No fitting model found. Check your parameters!")

    clients = []
    for i in range(fed_config["K"]):
        clients.append(Client(f"Client_{i + 1}"))
    server = Flame_server(model, fed_config, clients)

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
