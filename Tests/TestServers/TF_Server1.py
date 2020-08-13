import time

from Core.Backends import TensorFlow
from Core.Server import start_server
server = start_server("0.0.0.0:8901", TensorFlow.TFBackend(1))
time.sleep(10000)