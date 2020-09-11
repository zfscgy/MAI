from MAI.Core.Expression.VTensor import *


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


def conv2d(x, filters, strides=1):
    """
    :param x: shape of [..., height, width, channel]
    :param filters: shape of [size_h, size_w, in_channel, out_channel]
    :param strides: length-2 list or int
    :return: shape of [..., (height + 1 - size_h) / stride[1], (width + 1 - size_w)/stride[0], out_channel]
    """
    x = VTensor(x)
    filters = VTensor(filters)
    if isinstance(strides, int):
        strides = [strides, strides]
    if x.shape()[-1] != filters.shape()[-2]:
        raise ExpressionError(
            "The input shape %s is incompatible with filters shape %s" % (x.shape(), filters.shape())
        )
    if (x.shape()[-3] - filters.shape()[-3] + 1) % strides[0] != 0 or\
            (x.shape()[-2] - filters.shape()[-2] + 1) % strides[1] != 0:
        raise ExpressionError(
            "The stride %s is incompatible with input shape %s and filters shape %s" %
            (strides, x.shape(), filters.shape())
        )
    return ExprVTensor(
        x.shape()[:-3] +
        [
            (x.shape()[-3] - filters.shape()[-3] + 1) / strides[0],
            (x.shape()[-2] - filters.shape()[-2] + 1) / strides[1],
         filters.shape()[-1]
        ],
        "conv2d %s %s %s" % (x, filters, VTensor(strides))
    )


def conv2d_transpose(x, filters, strides=1):
    """
    It is the reverse of convolution.
    :param x: [..., height, width, channel]
    :param filters: [..., size_h, size_w, out_channel, in_channel]
    :param strides: [..., (height + size_h]
    :return:
    """
    x = VTensor(x)
    filters = VTensor(filters)
    if isinstance(strides, int):
        strides = [strides, strides]
    return ExprVTensor(
        x.shape()[:-3] + [(x.shape()[-2] - 1) * strides[0] + filters.shape()[0],
                          (x.shape()[-1] - 1) * strides[1] + filters.shape()[1],
                          filters.shape()[3]],
        "conv2d_t %s %s %s" % (x, filters, VTensor(strides))
    )


def avg_pool2d(x, kernel_size):
    """
    :param x:
    :param kernel_size:
    :return:
    """
    x = VTensor(x)
    if x.shape()[-3] % kernel_size[0] != 0 or x.shape()[-2] % kernel_size[1] != 0:
        raise ExpressionError("Input shape %s is incompatible with kernel shape %s" % (x.shape(), kernel_size))
    return ExprVTensor(
        x.shape()[:-3] + [int(x.shape()[-3] / kernel_size[0]), int(x.shape()[-2]) / kernel_size[1], x.shape()[-1]],
        "avg_pool2d %s %s" % (x, VTensor(kernel_size))
    )


def up_sample2d(x, scales):
    """
    :param x:
    :param scales:
    :return:
    """
    x = VTensor(x)
    if isinstance(scales, int):
        scales = [scales, scales]
    return ExprVTensor(
        x.shape()[:-3] + [scales[0] * x.shape()[-3], scales[1] * x.shape()[-2], x.shape()[-1]],
        "up_sample2d %s %s" % (x, VTensor(scales))
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