from MAI.Protocols.RTAS import RTAS
import MAI.Core.Expression.GenExpr as ge

rtas = RTAS("127.0.0.1:8900", "127.0.0.1:8901", "127.0.0.2:8902", "127.0.0.3:8903")

print("Start test RTAS \n[sigmoid]\n=======")
tensor0 = rtas.cluster.compute(ge.new([[1., 2.], [3., 4.]]), 0)
print("This should be [[1., 2.], [3., 4.]]\n", tensor0.reveal())
tensor1 = rtas.share(tensor0)
tensor2 = rtas.sigmoid(tensor1)
tensor3 = rtas.reconstruct(tensor2, 0)
print("This should be sigmoid([[1., 2.], [3., 4.]]):\n", tensor3.reveal())
print("Finished test RTAS [sigmoid]\n=========")