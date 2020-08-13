import Core.GenExpr as ge
from Protocols.ASharing import ASharing, ASharedTensor


class RTAS(ASharing):
    def __init__(self, server0, server1, server2, rt_producer: str, *args):
        super(RTAS, self).__init__(server0, server1, server2, rt_producer, *args)

    def _elementwise_function(self, shared_tensor: ASharedTensor, func):
        def calc_size(shape: list):
            size = 1
            for dim in shape:
                size *= dim
            return size
        raw_shape = shared_tensor.shape()
        random_permutation = self.cluster.compute(ge.random_permutation(size=calc_size(raw_shape)), 0)
        inverse_permutation = self.cluster.compute(ge.inverse_permutation(random_permutation), 0)
        random_permuted_shared_tensor = ASharedTensor(
            self.cluster.compute(ge.gather_1d(ge.reshape(shared_tensor.v0, [-1]), random_permutation), 0),
            self.cluster.compute(ge.gather_1d(ge.reshape(shared_tensor.v1, [-1]), random_permutation), 1)
        )
        permuted_tensor = self.reconstruct(random_permuted_shared_tensor, 3)
        elementwise_output = self.cluster.compute(func(permuted_tensor), 3)
        shared_output = self.share(elementwise_output)

        shared_output_recovered = ASharedTensor(
            self.cluster.compute(ge.reshape(ge.gather_1d(shared_output.v0, inverse_permutation), raw_shape), 0),
            self.cluster.compute(ge.reshape(ge.gather_1d(shared_output.v1, inverse_permutation), raw_shape), 1)
        )
        return shared_output_recovered

    def sigmoid(self, shared_tensor: ASharedTensor):
        return self._elementwise_function(shared_tensor, ge.sigmoid)

    def softmax(self, shared_tensor: ASharedTensor):
        return self._elementwise_function(shared_tensor, ge.softmax)

    def relu(self, shared_tensor: ASharedTensor):
        return self._elementwise_function(shared_tensor, ge.relu)