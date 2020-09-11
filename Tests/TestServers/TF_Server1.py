import time

from MAI.Core.Backends import TensorFlow
from MAI.Core.Cluster.Server import start_server
server = start_server("0.0.0.0:8901", TensorFlow.TFBackend(1, disable_gpu=True))
time.sleep(10000)