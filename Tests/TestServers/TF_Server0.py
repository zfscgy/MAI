import time

from Core.Backends import TensorFlow
from Core.Server import start_server

server = start_server("0.0.0.0:8900", TensorFlow.TFBackend(0))
time.sleep(10000)
