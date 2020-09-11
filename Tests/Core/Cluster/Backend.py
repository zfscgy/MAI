from MAI.Core.Cluster.Cluster import ClusterController
import MAI.Core.Expression.GenExpr as ge

cluster_controller = ClusterController(["127.0.0.1:8900"])


print("Start test \n[random_perm]\n[inv_perm]\n============")
tensor0 = cluster_controller.compute(ge.random_permutation(10), 0)
print("This should be a permutation of 1..10:\n", tensor0.reveal())
tensor1 = cluster_controller.compute(ge.inverse_permutation(tensor0), 0)
print("This should be the inverse of prior permutation:\n", tensor1.reveal())
print("Finished test [random_perm] [inv_perm]\n==========")