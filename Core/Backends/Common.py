import numpy as np
import pandas as pd
import traceback
import pickle

from Core.MPC_pb2_grpc import MPCServerServicer
import Core.MPC_pb2 as pb
from Core.Cluster import RemoteServer


def log(err):
    print(err)
    print(traceback.format_exc())


class BackendException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class Container:
    def __init__(self):
        self.value_dict = dict()
        self.cleared_indices = set()
        self.next_idx = 1

    def get(self, idx):
        return self.value_dict.get(idx)

    def store(self, val):
        if len(self.cleared_indices) != 0:
            idx = self.cleared_indices.pop()
        else:
            idx = self.next_idx
            self.next_idx += 1
        self.value_dict[idx] = val
        return idx

    def delete(self, next_idx):
        if next_idx in self.value_dict:
            del self.value_dict[next_idx]
            self.cleared_indices.add(next_idx)


N_Paras = {
    # Bsic operations
    "new": 1,
    "range": 2,
    "random_perm": 1,
    "inv_perm": 1,

    "set_loader_random_seed": 1,
    "load_data": 2,
    "load_batch": 2,


    # These depends on different backends, i.e. numpy, pytorch, tensorflow
    "add": 2,
    "mul": 2,
    "sub": 2,
    "matmul": 2,

    "gather_1d": 3,
    "reshape": 2,
    "transpose": 1,
    "sum": 2,
    "mean": 2,

    "random_normal": 3,
    "random_uniform": 3,
    "random_int": 3,

    "sigmoid": 1,
    "softmax": 1,
    "relu": 1,

    "std": 1,


}


class Backend(MPCServerServicer):
    def __init__(self, server_id, remote_servers=None):
        self.server_id = server_id
        self.tensor_container = Container()
        self.server_dict = {}
        if remote_servers is None:
            remote_servers = []
        for remote_server in remote_servers:
            self.server_dict[remote_server.server_id] = remote_server
        self.n_paras = N_Paras.copy()

        self.loader_random_generator = np.random.default_rng(np.random.randint(0, 999999))

        def set_loader_random_seed(seed):
            seed = int(self.to_numpy(seed))
            self.loader_random_generator = np.random.default_rng(seed)
            return None

        def random_permutation(size: int):
            perm = np.arange(size)
            np.random.shuffle(perm)
            return perm

        def inverse_permutation(perm: np.ndarray):
            inv_perm = perm.copy()
            for i, val in enumerate(perm):
                inv_perm[val] = i
            return inv_perm

        self.funcs = {
            "new": lambda x: x,
            "range": lambda low, high: np.arange(low, high),
            "random_perm": random_permutation,
            "inv_perm": inverse_permutation,
            "set_loader_random_seed": set_loader_random_seed,
            "load_data": lambda data_path, data_type: pd.read_csv(data_path, header=None).values.astype(data_type),
            "load_batch": lambda data, batch_size: self.loader_random_generator.choice(data, batch_size, axis=0)
        }

    def eval(self, x):
        return eval(x)

    def to_tensor(self, tensor):
        return tensor

    def to_numpy(self, tensor):
        return tensor

    def fetch_tensor(self, tensor_spec: str):
        server_id, idx = tensor_spec.split(":")
        server_id = int(server_id)
        idx = int(idx)
        if server_id == self.server_id:
            tensor = self.tensor_container.get(idx)
        else:
            tensor = self.server_dict[server_id].get_tensor(idx)
        if tensor is None:
            raise BackendException("Tensor %s not exist. It may be destroyed." % tensor_spec)
        return tensor

    def evaluate_expr(self, tokens: list):
        def is_tensor(token: str):
            return ":" in token

        def is_op(token: str):
            return token in self.funcs

        paras = []
        op = tokens[0]
        start = 1
        for _ in range(self.n_paras[op]):
            if is_op(tokens[start]):
                tensor, tokens_consumed = self.evaluate_expr(tokens[start:])
                paras.append(tensor)
                start += tokens_consumed
            elif is_tensor(tokens[start]):
                paras.append(self.fetch_tensor(tokens[start]))
                start += 1
            else:
                paras.append(self.eval(tokens[start]))
                start += 1
        return self.funcs[op](*paras), start

    def GetTensor(self, request, context):
        tensor = self.tensor_container.get(request.tensor_id)
        return pb.Tensor(tensor_buffer=pickle.dumps(self.to_numpy(tensor)))

    def Compute(self, request, context):
        try:
            tensor, _ = self.evaluate_expr(request.computation.split(" "))
            if tensor is not None:
                tensor_id = self.tensor_container.store(tensor)
                return pb.TensorSpec(tensor_id=tensor_id, shape=tensor.shape)
            else:
                return pb.TensorSpec(tensor_id=0, shape=[])
        except Exception as e:
            log("Exception while evaluating expression: " + str(e))
            return pb.TensorSpec(tensor_id=-1, shape=[])

    def DeleteTensor(self, request, context):
        self.tensor_container.delete(request.tensor_id)
        return pb.Status(res=0)

    def SetServerSpec(self, request, context):
        for server in request.servers:
            self.server_dict[server.id] = RemoteServer(server.id, server.address)
        return pb.Status(res=0)
