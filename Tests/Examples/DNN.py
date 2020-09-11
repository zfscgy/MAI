import numpy as np
import time
from MAI.Protocols.RTAS import RTAS, ASharedTensor
import MAI.Core.Expression.GenExpr as ge
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

tf.config.set_visible_devices([], 'GPU')

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
    batch_ys = rtas.local_transform(batch, ge.gather_1d, ge.range1d(in_dim, in_dim + label_dim), 1)
    return batch_xs, batch_ys


weight = rtas.cluster.compute(ge.random_normal([in_dim, out_dim], 0, 1 / in_dim), 0)
bias = rtas.cluster.compute(ge.random_normal([out_dim], 0, 1), 0)
weight_shared = rtas.share(weight)
bias_shared = rtas.share(bias)

local_model = tf.keras.Sequential([
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.SGD(0.1)


def dnn_forward(xs: ASharedTensor):
    lin_out = rtas.add(rtas.matmul(xs, weight_shared), bias_shared)
    lin_out = rtas.reconstruct(lin_out).reveal()
    return local_model(lin_out)


def dnn(xs: ASharedTensor, ys):
    global weight_shared, bias_shared
    lin_out = rtas.add(rtas.matmul(xs, weight_shared), bias_shared)
    lin_out = tf.Variable(rtas.reconstruct(lin_out).reveal())
    with tf.GradientTape(persistent=True) as tape:
        preds = local_model(lin_out)
        loss = tf.keras.losses.categorical_crossentropy(ys, preds)
    grads_on_model = tape.gradient(loss, local_model.trainable_variables)
    optimizer.apply_gradients(zip(grads_on_model, local_model.trainable_variables))
    grads_on_lin_out = rtas.share(rtas.cluster.set_tensor(tape.gradient(loss, lin_out), 0))
    grads_on_bias = rtas.local_transform(grads_on_lin_out, ge.reduce_sum, 0)
    grads_on_weights = rtas.matmul(rtas.local_transform(xs, ge.transpose), grads_on_lin_out)
    weight_shared = rtas.sub(weight_shared, rtas.mul(grads_on_weights, learning_rate / batch_size))
    bias_shared = rtas.sub(bias_shared, rtas.mul(grads_on_bias, learning_rate / batch_size))


n_rounds = 10001
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
        pred_ys = dnn_forward(xs)
        acc = accuracy(rtas.reconstruct(ys).reveal(), pred_ys)
        print("Time %.4f, ACC: %.4f" % (time.time() - start_time, acc))
        records.append([time.time() - start_time, acc])
    xs, ys = load_batch(train_data_shared, batch_size)
    dnn(xs, rtas.reconstruct(ys).reveal())

np.savetxt("logistic.csv", np.array(records), delimiter=",")