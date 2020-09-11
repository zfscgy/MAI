import time

from MAI.Core.Backends import TensorFlow
from MAI.Core.Cluster.Server import start_server

server = start_server("0.0.0.0:8900", TensorFlow.TFBackend(0, disable_gpu=True))
time.sleep(10000)
