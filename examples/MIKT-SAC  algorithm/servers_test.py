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

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        # just send back the same data, but upper-cased
        self.request.sendall(self.data.upper())

if __name__ == "__main__":
    HOST, PORT ="10.249.159.181", 3081
    # HOST, PORT = "10.249.156.0", 3100

    # Create the server, binding to localhost on port 9999
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()