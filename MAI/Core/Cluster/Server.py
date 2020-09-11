import grpc
import concurrent.futures as futures
from MAI.Core.Cluster.MPC_pb2_grpc import MPCServerServicer, add_MPCServerServicer_to_server

server = None

def start_server(port, servicer: MPCServerServicer):
    global server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=3),
                         options=[('grpc.max_receive_message_length', 256 * 1024 * 1024)])
    server.add_insecure_port(port)
    add_MPCServerServicer_to_server(servicer, server)
    server.start()
    return server

def stop_server():
    if server:
        server.stop()