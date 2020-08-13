import tensorflow as tf
import numpy as np
from Core.Backends.Common import Backend, BackendException


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


class TFBackend(Backend):
    def __init__(self, server_id, remote_servers=None):
        super(TFBackend, self).__init__(server_id, remote_servers)
        self.funcs.update({
            "add": tf.add,
            "mul": tf.multiply,
            "sub": tf.subtract,
            "matmul": tf.matmul,

            "gather_1d": lambda x, y, axis: tf.gather(x, y, axis=axis),
            "reshape": lambda x, s: tf.reshape(x, s.tolist()),
            "transpose": tf.transpose,

            "sum": tf.reduce_sum,
            "mean": tf.reduce_mean,

            "random_normal": lambda
                shape, mean, std: tf.random.normal(shape.tolist(), mean, std),
            "random_uniform": lambda
                shape, low, high: tf.random.normal(shape.tolist(), low, high),
            "random_int": lambda shape, low, high: tf.random.uniform(shape.tolist(), low, high + 1, tf.int32),
            "std": tf.math.reduce_std,



            "sigmoid": tf.sigmoid,
            "softmax": tf.nn.softmax,
            "relu": tf.nn.relu
        })

    def to_numpy(self, tensor):
        if tensor is None:
            return tensor
        elif isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, tf.Tensor):
            return np.array(tensor.numpy())
        else:
            raise BackendException("Cannot conver %s to numpy, type: %s" % (tensor, type(tensor)))

    def to_tensor(self, tensor):
        return tensor

    def eval(self, x):
        x = eval(x)
        if isinstance(x, float):
            return np.array(x).astype('float32')
        elif isinstance(x, int) or isinstance(x, str):
            return x
        else:
            x = np.array(x)
            if x.dtype in [np.float or np.dtype('float64')]:
                return x.astype(np.float32)
            return x.astype(np.int)
