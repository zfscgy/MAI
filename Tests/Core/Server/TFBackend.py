from Core.Cluster import ClusterController
import Core.GenExpr as ge

cluster_controller = ClusterController(["127.0.0.1:8900"])


print("Start test \n[new]\n[gather_1d]\n[reshape]\n============")
tensor0 = cluster_controller.compute(ge.new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 0)
print("This should be [[1, 2, 3], [4, 5, 6], [7, 8, 9]]:\n", tensor0.reveal())
tensor1 = cluster_controller.compute(ge.gather_1d(tensor0, [1, 2]), 0)
print("This should be [[4, 5, 6], [7, 8, 9]]:\n", tensor1.reveal())
tensor2 = cluster_controller.compute(ge.reshape(tensor1, [6]), 0)
print("This should be [4, 5, 6, 7, 8, 9]:\n", tensor2.reveal())
print("Finished test [new] [gather_1d] [reshape]\n==========")
print("Start test \n[sum]\n[mean]\n===========")
tensor3 = cluster_controller.compute(ge.reduce_sum(tensor2), 0)
print("This should be 39:", tensor3.reveal())
tensor4 = cluster_controller.compute(ge.reduce_mean(tensor0, axis=1), 0)
print("This should be [6, 15, 24]", tensor4.reveal())
print("Finished test [sum] [mean]\n===========")
