import time

from Core.Backends import TensorFlow
from Core.Server import start_server

server = start_server("0.0.0.0:8903", TensorFlow.TFBackend(2))
time.sleep(10000)