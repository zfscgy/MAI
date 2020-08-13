import grpc
import Core.MPC_pb2 as pb
from Core.MPC_pb2_grpc import MPCServerStub
import Core.GenExpr as ge
import pickle


class RPCException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class RemoteServer:
    def __init__(self, server_id, address):
        self.server_id = server_id
        channel = grpc.insecure_channel(address, options=[('grpc.max_receive_message_length', 256 * 1024 * 1024)])
        self.stub = MPCServerStub(channel)

    def set_server_spec(self, server_addrs):
        server_spec = pb.ServerSpec(servers=[])
        for i, server_addr in enumerate(server_addrs):
            server_spec.servers.append(pb.ServerDef(id=i, address=server_addr))
        status = self.stub.SetServerSpec(server_spec)
        if status.res != 0:
            raise RPCException("Remote server return not 0 status")

    def get_tensor(self, tensor_id):
        return pickle.loads(self.stub.GetTensor(pb.TensorSpec(tensor_id=tensor_id)).tensor_buffer)

    def delete_tensor(self, tensor_id):
        status = self.stub.DeleteTensor(pb.TensorSpec(tensor_id=tensor_id))
        if status.res != 0:
            raise RPCException("Remote server return not 0 status")

    def compute(self, computation: str):
        tensor = self.stub.Compute(pb.Computation(computation=computation))
        if tensor.tensor_id == -1:
            raise Exception("Failed to do computation: " + computation)
        return tensor.tensor_id, list(tensor.shape)


class RemoteTensor(ge.VTensor):
    def __init__(self, server: RemoteServer, tensor_id: int, shape):
        super(RemoteTensor, self).__init__()
        self.server = server
        self.tensor_id = tensor_id
        self.tensor_shape = shape
        self.comp_str = "%d:%d" % (self.server.server_id, self.tensor_id)

    def reveal(self):
        return self.server.get_tensor(self.tensor_id)

    def __del__(self):
        return self.server.delete_tensor(self.tensor_id)

    def __str__(self):
        return "RemoteTensor at server %d: index %d, with shape: %s" % \
               (self.server.server_id, self.tensor_id, self.shape())


class ClusterController:
    def __init__(self, server_addrs: list):
        self.remote_servers = dict()
        for i, addr in enumerate(server_addrs):
            self.remote_servers[i] = RemoteServer(i, addr)
        for i in range(len(server_addrs)):
            self.remote_servers[i].set_server_spec(server_addrs)

    def compute(self, computation: ge.ExprVTensor, server_id):
        if server_id not in self.remote_servers:
            raise RPCException("Remote server with id %d not exist" % server_id)
        tensor_id, shape = self.remote_servers[server_id].compute(str(computation))
        return RemoteTensor(self.remote_servers[server_id], tensor_id, shape)

