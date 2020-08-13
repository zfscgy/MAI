import Core.GenExpr as ge
from Core.Cluster import ClusterController
from Protocols.ASharing import ASharing

protocol = ASharing(
    "127.0.0.1: 8900",
    "127.0.0.1: 8901",
    "127.0.0.2: 8902"
)

print("Start test \n[add]\n[share]\n[matmul]\n[mul]\n===========")
tensor0 = protocol.cluster.compute(ge.add([[1., 2.]], [[3., 4.]]), 0)
print("This should be [4., 6.]:", tensor0.reveal())
tensor0_shared = protocol.share(tensor0)
print("This should be two random tensors with sum [4., 6.]: ", tensor0_shared.v0.reveal(), tensor0_shared.v1.reveal())
tensor1 = protocol.cluster.compute(ge.new([[3.], [5.]]), 1)
tensor1_shared = protocol.share(tensor1)
shared_product12 = protocol.matmul(tensor0_shared, tensor1_shared)
print("This should be [[42]]", protocol.reconstruct(shared_product12).reveal())
tensor1_square = protocol.mul(tensor1_shared, tensor1_shared)
print("This should be [[9], [25]]:\n", protocol.reconstruct(tensor1_square).reveal())

print("Finished test [add] [share] [matmul] [mul]\n============")

print("Start test \n[load_batch]\n==========")
tensor2 = protocol.cluster.compute(ge.load_batch([1, 2, 3, 4, 5, 6, 7], 5), 0)
tensor3 = protocol.cluster.compute(ge.load_batch([1, 2, 3, 4, 5, 6, 7], 5), 1)
print("This should be two same tensors:", tensor2.reveal(), tensor3.reveal())
print("Finished test [load_batch]\n=========")