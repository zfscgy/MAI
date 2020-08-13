from Core.Cluster import ClusterController, RemoteTensor
import Core.GenExpr as ge


class ASharingError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class ASharedTensor:
    def __init__(self, v0: RemoteTensor, v1: RemoteTensor):
        if v0.shape() != v1.shape():
            raise ASharingError("Shape not match: %s, %s" % (v0.shape(), v1.shape()))
        self.v0 = v0
        self.v1 = v1

    def shape(self):
        return self.v0.shape()

    def __str__(self):
        return "AShared tensor with shape: %s" % self.shape()


class ASharing:
    def __init__(self, addr_0: str, addr_1: str, addr_2: str, *args):
        self.cluster = ClusterController([addr_0, addr_1, addr_2] + list(args))
        random_seed = self.cluster.compute(
            ge.random_int([], 0, 999999), 0
        )
        self.cluster.compute(
            ge.set_loader_random_seed(random_seed), 0
        )
        self.cluster.compute(
            ge.set_loader_random_seed(random_seed), 1
        )

    def share(self, remote_tensor: RemoteTensor):
        v0_raw = self.cluster.compute(
            ge.mul(ge.random_normal(remote_tensor.shape(), 0, ge.std(remote_tensor)), remote_tensor),
            remote_tensor.server.server_id)
        v1_raw = self.cluster.compute(ge.sub(remote_tensor, v0_raw), remote_tensor.server.server_id)
        v0 = self.cluster.compute(ge.new(v0_raw), 0)
        v1 = self.cluster.compute(ge.new(v1_raw), 1)
        return ASharedTensor(v0, v1)

    def reconstruct(self, shared_tensor: ASharedTensor, to_server=0):
        return self.cluster.compute(
            ge.add(shared_tensor.v0, shared_tensor.v1), to_server
        )

    def _linear(self, left, right, func):
        if isinstance(left, ASharedTensor):
            if isinstance(right, ASharedTensor):
                return ASharedTensor(
                    self.cluster.compute(func(left.v0, right.v0), 0),
                    self.cluster.compute(func(left.v1, right.v1), 1)
                )
            else:
                return ASharedTensor(
                    self.cluster.compute(func(left.v0, right), 0),
                    self.cluster.compute(func(left.v1, right), 1)
                )
        else:
            if isinstance(right, ASharedTensor):
                return ASharedTensor(
                    self.cluster.compute(func(ge.mul(left, 1/2), right.v0), 0),
                    self.cluster.compute(func(ge.mul(left, 1/2), right.v1), 0)
                )
            else:
                return self.cluster.compute(func(left, right), 0)

    def add(self, left, right):
        return self._linear(left, right, ge.add)
    
    def sub(self, left, right):
        return self._linear(left, right, ge.sub)

    def _product(self, left, right, func):
        if isinstance(left, ASharedTensor):
            if isinstance(right, ASharedTensor):
                u = self.cluster.compute(ge.random_normal(left.shape(), 0, ge.std(left.v0)), 2)
                v = self.cluster.compute(ge.random_normal(right.shape(), 0, ge.std(right.v0)), 2)
                w = self.cluster.compute(func(u, v), 2)
                u_shared = self.share(u)
                v_shared = self.share(v)
                w_shared = self.share(w)
                l_sub_u = self.reconstruct(self.sub(left, u_shared))
                r_sub_v = self.reconstruct(self.sub(right, v_shared))
                # (l-u)(r-v)
                mul0 = self.cluster.compute(func(l_sub_u, r_sub_v), 0)
                # (l-u)v
                mul1_0 = self.cluster.compute(func(l_sub_u, v_shared.v0), 0)
                mul1_1 = self.cluster.compute(func(l_sub_u, v_shared.v1), 1)
                # u(r-v)
                mul2_0 = self.cluster.compute(func(u_shared.v0, r_sub_v), 0)
                mul2_1 = self.cluster.compute(func(u_shared.v1, r_sub_v), 1)
                return self.add(
                    ASharedTensor(
                        self.cluster.compute(ge.add(ge.add(mul0, mul1_0), mul2_0), 0),
                        self.cluster.compute(ge.add(mul1_1, mul2_1), 1)
                    ),
                    w_shared
                )
            else:
                return ASharedTensor(
                    self.cluster.compute(func(left.v0, right), 0),
                    self.cluster.compute(func(left.v1, right), 1)
                )
        else:
            if isinstance(right, ASharedTensor):
                return ASharedTensor(
                    self.cluster.compute(func(left, right.v0), 0),
                    self.cluster.compute(func(left, right.v1), 1)
                )
            else:
                return self.cluster.compute(func(left, right), 0)

    def mul(self, left, right):
        return self._product(left, right, ge.mul)

    def matmul(self, left, right):
        return self._product(left, right, ge.matmul)

    def local_transform(self, x: ASharedTensor, transform, *args):
        return ASharedTensor(
            self.cluster.compute(transform(x.v0, *args), 0),
            self.cluster.compute(transform(x.v1, *args), 1)
        )