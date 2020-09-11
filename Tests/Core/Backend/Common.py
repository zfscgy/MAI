from MAI.Core.Backends.Common import Backend

backend = Backend(0)

print("Start tests \n[new]\n[random_perm]\n[load_data]\n[set_loader_random_seed]\n=======")

tensor1, _ = backend.evaluate_expr("new 9999".split(' '))  # 1
print("This should be 9999: ", tensor1)
tensor2, _ = backend.evaluate_expr("random_perm 10".split(' '))  # 2
print("This should be a random permutation of 1..10", tensor2)
tensor3, _ = backend.evaluate_expr("load_data 'Tests/TestServers/mnist.csv' 'float32'".split(' '))
print("This should be a shape [55000, 794]: ", tensor3.shape)
tensor4, _ = backend.evaluate_expr("set_loader_random_seed 1024".split(' '))
print("This should be None: ", tensor4)

print("Tests [new] [random_perm] [load_data] [set_loader_random_seed] finished\n======")
print("Start tests \n[load_batch]\n==============")
tensor5, _ = backend.evaluate_expr("load_batch [[1,2,3],[4,5,6]] 1".split(' '))
print("This should be [[1, 2, 3]] or [[4, 5, 6]]:", tensor5)
print("Finished tests [load_batch]\n==========")