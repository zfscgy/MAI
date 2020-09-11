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
