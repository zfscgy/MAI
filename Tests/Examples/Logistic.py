import numpy as np
import time
import matplotlib.pyplot as plt
from Protocols.RTAS import RTAS, ASharedTensor
import Core.GenExpr as ge


def accuracy(ys, pred_ys):
    return np.mean(np.argmax(pred_ys, axis=1) == np.argmax(ys, axis=1))

rtas = RTAS("127.0.0.1:8900", "127.0.0.1:8901", "127.0.0.2:8902", "127.0.0.3:8903")


in_dim = 784
out_dim = 10
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
    batch_ys = rtas.local_transform(batch, ge.gather_1d, ge.range1d(in_dim, in_dim + out_dim), 1)
    return batch_xs, batch_ys


weight = rtas.cluster.compute(ge.random_normal([in_dim, out_dim], 0, 1 / in_dim), 0)
bias = rtas.cluster.compute(ge.random_normal([out_dim], 0, 1), 0)
weight_shared = rtas.share(weight)
bias_shared = rtas.share(bias)


def logistic(xs: ASharedTensor):
    return rtas.sigmoid(rtas.add(rtas.matmul(xs, weight_shared), bias_shared))


def logistic_backward(xs: ASharedTensor, ys: ASharedTensor, preds: ASharedTensor):
    grads_on_pred = rtas.mul(2, rtas.sub(preds, ys))
    grads_on_linear_out = rtas.mul(grads_on_pred, rtas.mul(preds, rtas.sub(1., preds)))
    # grads_on_linear_out = rtas.sub(preds, ys)
    grads_on_bias = rtas.local_transform(grads_on_linear_out, ge.reduce_sum, 0)
    grads_on_weights = rtas.matmul(rtas.local_transform(xs, ge.transpose), grads_on_linear_out)
    return grads_on_bias, grads_on_weights


n_rounds = 10000
batch_size = 32
test_batch_size = 5000
test_per_batches = 100
learning_rate = 0.1
records = []
start_time = time.time()
for i in range(n_rounds):
    print("Round %d" % i)
    if i % test_per_batches == 0:
        xs, ys = load_batch(test_data_shared, test_batch_size)
        pred_ys = logistic(xs)
        ys_revealed = rtas.reconstruct(ys).reveal()
        pred_ys_revealed = rtas.reconstruct(pred_ys).reveal()
        acc = accuracy(ys_revealed, pred_ys_revealed)
        print("Time %.4f, ACC: %.4f" % (time.time() - start_time, acc))
        records.append([time.time() - start_time])
    xs, ys = load_batch(train_data_shared, batch_size)
    pred_ys = logistic(xs)
    g_bias, g_weights = logistic_backward(xs, ys, pred_ys)
    weight_shared = rtas.sub(weight_shared, rtas.mul(g_weights, learning_rate / batch_size))
    bias_shared = rtas.sub(bias_shared, rtas.mul(g_bias, learning_rate / batch_size))

np.savetxt("logistic.csv", np.array(records), delimiter=",")