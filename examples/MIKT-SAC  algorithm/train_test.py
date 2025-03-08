from multiprocessing import Process
import time
import argparse
import socket
import os
from utils import check_ports_avail
from RemoteEnvironmentServer import RemoteEnvironmentServer

from hanshu.pendulum import PendulumEnv
import numpy as np

import gym
import pandas as pd
import torch
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import socketserver

import socket
import sys

HOST, PORT = "10.249.159.181", 3081
# HOST, PORT = "LPF", 3100
# HOST, PORT = "10.249.156.0", 3100
data = " ".join(sys.argv[1:])

# Create a socket (SOCK_STREAM means a TCP socket)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    # Connect to server and send data
    sock.connect((HOST, PORT))
    sock.sendall(bytes(data + "\n", "utf-8"))

    # Receive data from the server and shut down
    received = str(sock.recv(1024), "utf-8")

print("Sent:     {}".format(data))
print("Received: {}".format(received))