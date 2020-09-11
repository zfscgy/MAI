import numpy as np
from MAI.Protocols.RTAS import RTAS, ASharedTensor
import MAI.Core.Expression.GenExpr as ge
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')


def accuracy(ys, pred_ys):
    return np.mean(np.argmax(pred_ys, axis=1) == np.argmax(ys, axis=1))


rtas = RTAS("127.0.0.1:8900", "127.0.0.1:8901", "127.0.0.2:8902", "127.0.0.3:8903")


in_dim = 784
out_dim = 64
label_dim = 10
train_size = 50000
test_size = 5000
scale = 0.1
data_file = 'mnist.csv'

data = rtas.cluster.compute(ge.load_data(data_file), 0)
train_data = rtas.cluster.compute(ge.gather_1d(data, ge.range1d(0, train_size)), 0)
test_data = rtas.cluster.compute(ge.gather_1d(data, ge.range1d(train_size, train_size + test_size)), 0)
train_data_shared = rtas.share(train_data)
test_data_shared = rtas.share(test_data)


def load_batch(data, batch_size: int):
    batch = rtas.local_transform(data, ge.load_batch, batch_size)
    batch_xs = rtas.local_transform(batch, ge.gather_1d, ge.range1d(0, in_dim), 1)
    batch_xs = rtas.mul(batch_xs, scale)
    batch_xs = rtas.local_transform(batch_xs, ge.reshape, [-1, 32, 32, 1])
    batch_ys = rtas.local_transform(batch, ge.gather_1d, ge.range1d(in_dim, in_dim + label_dim), 1)
    return batch_xs, batch_ys


class Lenet:
    def __init__(self):
        self.filter_1_shared = rtas.share(rtas.cluster.compute(ge.random_normal([5, 5, 1, 6], 0, 0.3)))
        self.avg_pool_1 = rtas.cluster.compute(ge.reshape(ge.new([]), []))