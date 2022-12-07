import os
import time
import torch
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from config import SAVE_PATH, device
from utils import get_data_by_indices
from client import Client
import logging
from random import random
from matplotlib.ticker import MaxNLocator

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Adv_client(Client): #TODO: Implement adersary
    def __init__(self, name):
        super().__init__(name)