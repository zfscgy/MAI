import numpy as np


class ExpressionError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class VTensor:
    def __init__(self, x=None):
        if x is None:
            pass
        elif type(x) in [int, float]:
            self.tensor_shape = []
            self.comp_str = str(x)
        elif type(x) == list:
            self.tensor_shape = list(np.array(x).shape)
            self.comp_str = str(x).replace(' ', '')
        else:
            if issubclass(type(x), VTensor):
                self.tensor_shape = x.tensor_shape
                self.comp_str = x.comp_str
            else:
                raise ExpressionError("Cannot convert to VTensor:" + str(x))

    def shape(self):
        return self.tensor_shape

    def __str__(self):
        return self.comp_str


class ExprVTensor(VTensor):
    def __init__(self, shape, comp_str: str):
        super(ExprVTensor, self).__init__()
        self.tensor_shape = shape
        self.comp_str = comp_str


def _can_broadcast(s1: list, s2: list):
    if s1 in [[], [1]] or s2 in [[], [1]]:
        return True
    min_len = min(len(s1), len(s2))
    if s1[-min_len:] == s2[-min_len:]:
        return True
    return False


def _broadcast_shape(s1: list, s2: list):
    return [s1, s2][int(len(s1) < len(s2))]


def _reduced_shape(shape: list, axis: list):
    new_shape = []
    for dim in shape:
        if dim not in axis:
            new_shape.append(dim)
    return new_shape


def new(x):
    x = VTensor(x)
    return ExprVTensor(
        x.shape(),
        "new %s" % VTensor(x)
    )


def range1d(low, high):
    low = VTensor(low)
    high = VTensor(high)
    return ExprVTensor(
        [-1],
        "range %s %s" % (low, high),
    )


def random_permutation(size):
    size = VTensor(size)
    return ExprVTensor(
        [size],
        "random_perm %s" % size
    )


def inverse_permutation(perm):
    perm = VTensor(perm)
    return ExprVTensor(
        perm.shape(),
        "inv_perm %s" % perm
    )


def add(left, right):
    left = VTensor(left)
    right = VTensor(right)
    if not _can_broadcast(left.shape(), right.shape()):
        raise ExpressionError("Cannot add with shapes %s, %s" % (left.shape(), right.shape()))
    return ExprVTensor(
        _broadcast_shape(left.shape(), right.shape()),
        "add %s %s" % (left, right)
    )


def sub(left, right):
    left = VTensor(left)
    right = VTensor(right)
    if not _can_broadcast(left.shape(), right.shape()):
        raise ExpressionError("Cannot sub with shapes %s, %s" % (left.shape(), right.shape()))
    return ExprVTensor(
        _broadcast_shape(left.shape(), right.shape()),
        "sub %s %s" % (left, right)
    )


def mul(left, right):
    left = VTensor(left)
    right = VTensor(right)
    if not _can_broadcast(left.shape(), right.shape()):
        raise ExpressionError("Cannot mul with shapes %s, %s" % (left.shape(), right.shape()))
    return ExprVTensor(
        _broadcast_shape(left.shape(), right.shape()),
        "mul %s %s" % (left, right)
    )


def matmul(left, right):
    left = VTensor(left)
    right = VTensor(right)
    if len(left.shape()) < 2 or len(right.shape()) != 2 or left.shape()[-1] != right.shape()[0]:
        raise ExpressionError("Cannot matmul with shapes %s, %s" % (left.shape(), right.shape()))
    return ExprVTensor(
        left.shape()[:-1] + [right.shape()[1]],
        "matmul %s %s" % (left, right)
    )


def gather_1d(x, indices, axis=0):
    x = VTensor(x)
    indices = VTensor(indices)
    return ExprVTensor(
        x.shape()[:axis] + indices.shape() + x.shape()[axis + 1:],
        "gather_1d %s %s %s" % (x, indices, VTensor(axis))
    )


def reshape(x, shape: list=None):
    x = VTensor(x)
    return ExprVTensor(
        shape,
        "reshape %s %s" % (x, VTensor(shape))
    )


def transpose(x):
    x = VTensor(x)
    return ExprVTensor(
        x.shape()[:-2] + x.shape()[-2:][::-1],
        "transpose %s" % x
    )


def reduce_sum(x, axis=None):
    if axis is None:
        axis = [i for i in range(len(x.shape()))]
    if isinstance(axis, int):
        axis = [axis]
    x = VTensor(x)
    return ExprVTensor(
        _reduced_shape(x.shape(), axis),
        "sum %s %s" % (x, VTensor(axis))
    )


def reduce_mean(x, axis=None):
    x = VTensor(x)
    if axis is None:
        axis = [i for i in range(len(x.shape()))]
    if isinstance(axis, int):
        axis = [axis]
    return ExprVTensor(
        _reduced_shape(x.shape(), axis),
        "sum %s %s" % (x, VTensor(axis))
    )


def random_normal(shape, mean, std):
    shape_str = str(shape).replace(' ', '')
    return ExprVTensor(
        shape,
        "random_normal %s %s %s" % (shape_str, mean, std)
    )


def random_uniform(shape, low, high):
    shape_str = str(shape).replace(' ', '')
    return ExprVTensor(
        shape,
        "random_uniform %s %f %f" % (shape_str, low, high)
    )


def random_int(shape, low, high):
    shape_str = str(shape).replace(' ', '')
    return ExprVTensor(
        shape,
        "random_int %s %d %d" % (shape_str, low, high)
    )


def sigmoid(x):
    x = VTensor(x)
    return ExprVTensor(
        x.shape(),
        "sigmoid %s" % x
    )


def softmax(x):
    x = VTensor(x)
    return ExprVTensor(
        x.shape(),
        "softmax %s" % x
    )


def relu(x):
    x = VTensor(x)
    return ExprVTensor(
        x.shape(),
        "relu %s" % x
    )


def std(x):
    x = VTensor(x)
    return ExprVTensor(
        [],
        "std %s" % x
    )


def set_loader_random_seed(x: VTensor):
    x = VTensor(x)
    return ExprVTensor(
        x.shape(),
        "set_loader_random_seed %s" % x
    )


def load_data(x: str, dtype: str = 'float32'):
    if ' ' in x:
        raise ExpressionError("Cannot have space in data path")
    return ExprVTensor(
        [-1],
        "load_data '%s' '%s'" % (x, dtype)
    )


def load_batch(x, batch_size: int):
    x = VTensor(x)
    return ExprVTensor(
        [batch_size] + x.shape()[1:],
        "load_batch %s %d" % (x, batch_size)
    )