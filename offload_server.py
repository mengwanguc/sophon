import multiprocessing as mp
from concurrent import futures
import grpc
import data_transfer_pb2
import data_transfer_pb2_grpc
import torch
from torchvision import datasets, transforms
import numpy as np
import sys

from PIL import Image
import io

import argparse

# https://pytorch.org/vision/stable/transforms.html
# python -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/data_transfer.proto

class PProcInfo():
    def __init__(self, pproc, size):
        self.pproc = pproc
        self.size  = size

# Hash of tensor object -> PProcInfo with smallest size
pproc_config = {}

# The following class implements the data transfering servie
class DataTransferService(data_transfer_pb2_grpc.DataTransferServicer):
    def __init__(self, num_workers=4, transform_all=transforms.Compose([])):
        '''
        num_workers: Number of workers performing preprocessing
        '''
        self.pool = futures.ThreadPoolExecutor(max_workers=num_workers)
        # self.pool = futures.ProcessPoolExecutor(max_workers=num_workers)
        self.transform_all = transform_all

    def get_data_nooff(self, request, context):
        path = request.fname
        response = data_transfer_pb2.DataResponse()
        with open(path, 'rb') as f:
            raw_data = f.read()
            response.image = raw_data
            response.fname = path
            response.pproc = 0
        return response
    
    # def get_data_alloff(self, request, context):
    #     path = request.fname
    #     response = data_transfer_pb2.DataResponse()
    #     # print(f"alloff request for {path}")
    #     with open(path, 'rb') as f:
    #         # read
    #         raw_data = f.read()
    #         # decode
    #         img = Image.open(io.BytesIO(raw_data))
    #         sample = img.convert('RGB')
    #         # all transform
    #         # print(self.transform_all)
    #         sample = self.transform_all(sample)

    #         response.image = sample.numpy().tobytes()
    #         response.fname = path
    #         response.pproc = 0
    #     # print(f"done getting response for {path}")
    #     return response
    
    def get_data_alloff(self, request, context):
        path = request.fname
        response = data_transfer_pb2.DataResponse()
        # print(f"alloff request for {path}")
        with open(path, 'rb') as f:
            # read
            raw_data = f.read()

            future = self.pool.submit(process_data_all, raw_data, self.transform_all)
            sample = future.result()  # Wait for the process to complete

            response.image = sample.numpy().tobytes()
            response.fname = path
            response.pproc = 0
            response.shape.extend(list(sample.shape))
            response.dtype = str(sample.numpy().dtype)
        # print(f"done getting response for {path}")
        return response
    
    # def get_data_alloff(self, request, context):
    #     path = request.fname
    #     response = data_transfer_pb2.DataResponse()
    #     # print(f"alloff request for {path}")

    #     future = self.pool.submit(process_data_all_entire, path, self.transform_all)
    #     sample = future.result()  # Wait for the process to complete

    #     response.image = sample.numpy().tobytes()
    #     response.fname = path
    #     response.pproc = 0
    #     response.shape.extend(list(sample.shape))
    #     response.dtype = str(sample.numpy().dtype)
    #     # print(f"done getting response for {path}")
    #     return response

    # def get_data_piloff(self, request, context):
    #     path = request.fname
    #     response = data_transfer_pb2.DataResponse()
    #     # print(f"alloff request for {path}")
    #     with open(path, 'rb') as f:
    #         # read
    #         raw_data = f.read()
    #         # decode
    #         img = Image.open(io.BytesIO(raw_data))
    #         sample = img.convert('RGB')
    #         # all transform
    #         # print(self.transform_all)
    #         for i in range(2):
    #             sample = self.transform_all.transforms[i](sample)

    #         response.image = sample.tobytes()
    #         response.pil_mode = sample.mode
    #         response.width = sample.size[0]
    #         response.height = sample.size[1]
    #         response.fname = path
    #         response.pproc = 0
    #     # print(f"done getting response for {path}")
    #     return response
    
    def get_data_piloff(self, request, context):
        path = request.fname
        response = data_transfer_pb2.DataResponse()
        # print(f"alloff request for {path}")
        with open(path, 'rb') as f:
            # read
            raw_data = f.read()
            

            future = self.pool.submit(process_data_pil, raw_data, self.transform_all)
            sample = future.result()  # Wait for the process to complete

            # print('get samle')
            response.image = sample.tobytes()
            response.pil_mode = sample.mode
            response.width = sample.size[0]
            response.height = sample.size[1]
            response.fname = path
            response.pproc = 0
        # print(f"done getting response for {path}")
        return response


    # This implementation supports all offloaded operations,
    # and sends performs preprocessing to another thread pool to control used core #.

    def get_data_smartoff(self, request, context):
        path = request.fname
        response = data_transfer_pb2.DataResponse()
        # print(f"alloff request for {path}")
        with open(path, 'rb') as f:
            # read
            sample = f.read()
            
            if request.pproc >= 0:
                future = self.pool.submit(process_data_smart, sample, self.transform_all, request.pproc)
                # sample = process_data_smart(sample, self.transform_all, request.pproc)
                sample = future.result()
            
            response.fname = path
            response.pproc = 0
            
            sample_type_name = type(sample).__name__
            if sample_type_name == 'bytes':
                # print('if bytes')
                response.image = sample
            elif sample_type_name == 'Image':
                # print('if PIL')
                response.image = sample.tobytes()
                response.pil_mode = sample.mode
                response.width = sample.size[0]
                response.height = sample.size[1]
            elif torch.is_tensor(sample):
                # print('if tensor')
                response.image = sample.numpy().tobytes()
                response.shape.extend(list(sample.shape))
                response.dtype = str(sample.numpy().dtype)
            else:
                # print("pproc {} not supported by smartoff".format(request.pproc))
                raise NotImplementedError("pproc {} not supported by smartoff".format(request.pproc))
        # print(f"done getting response for {path}")
        return response


    # # This implementation supports all offloaded operations, but performs preprocessing inside
    # # the grpc thread.
    # def get_data_smartoff(self, request, context):
    #     path = request.fname
    #     response = data_transfer_pb2.DataResponse()
    #     # print(f"alloff request for {path}")
    #     with open(path, 'rb') as f:
    #         # read
    #         sample = f.read()
    #         if request.pproc >= 0:
    #             sample = Image.open(io.BytesIO(sample))
    #             sample = sample.convert('RGB')

    #             for i in range(request.pproc+1):
    #                 sample = self.transform_all.transforms[i](sample)
            
    #         response.fname = path
    #         response.pproc = 0

    #         sample_type_name = type(sample).__name__
    #         if sample_type_name == 'bytes':
    #             # print('if bytes')
    #             response.image = sample
    #         elif sample_type_name == 'Image':
    #             # print('if PIL')
    #             response.image = sample.tobytes()
    #             response.pil_mode = sample.mode
    #             response.width = sample.size[0]
    #             response.height = sample.size[1]
    #         elif torch.is_tensor(sample):
    #             # print('if tensor')
    #             response.image = sample.numpy().tobytes()
    #         else:
    #             # print("pproc {} not supported by smartoff".format(request.pproc))
    #             raise NotImplementedError("pproc {} not supported by smartoff".format(request.pproc))
    #     # print(f"done getting response for {path}")
    #     return response

    # # This implementation only chooses from nooff and piloff
    # def get_data_smartoff(self, request, context):
    #     path = request.fname
    #     response = data_transfer_pb2.DataResponse()
    #     # print(f"alloff request for {path}")
    #     with open(path, 'rb') as f:
    #         # read
    #         raw_data = f.read()
    #         if request.pproc == -1:
    #             response.image = raw_data
    #             response.fname = path
    #             response.pproc = 0
    #         elif request.pproc == 1:
    #             # decode
    #             img = Image.open(io.BytesIO(raw_data))
    #             sample = img.convert('RGB')
    #             # all transform
    #             # print(self.transform_all)
    #             for i in range(2):
    #                 sample = self.transform_all.transforms[i](sample)

    #             response.image = sample.tobytes()
    #             response.pil_mode = sample.mode
    #             response.width = sample.size[0]
    #             response.height = sample.size[1]
    #             response.fname = path
    #             response.pproc = 0
    #         else:
    #             raise NotImplementedError("pproc {} not supported by smartoff".format(pproc))
    #     # print(f"done getting response for {path}")
    #     return response
    
def process_data_all_entire(path, transform_all):
    with open(path, 'rb') as f:
        # read
        raw_data = f.read()
        # decode
        img = Image.open(io.BytesIO(raw_data))
        sample = img.convert('RGB')
        # all transform
        sample = transform_all(sample)
        return sample
    

def process_data_all(raw_data, transform_all):
    # decode
    img = Image.open(io.BytesIO(raw_data))
    sample = img.convert('RGB')
    # all transform
    sample = transform_all(sample)
    return sample

def process_data_pil(raw_data, transform_all):
    # decode
    img = Image.open(io.BytesIO(raw_data))
    sample = img.convert('RGB')
    # all transform
    # print(self.transform_all)
    for i in range(2):
        sample = transform_all.transforms[i](sample)
    return sample

def process_data_smart(sample, transform_all, pproc):
    if pproc >= 0:
        sample = Image.open(io.BytesIO(sample))
        sample = sample.convert('RGB')

        for i in range(pproc):
            sample = transform_all.transforms[i](sample)
    return sample




parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--grpc-workers', default=32, type=int,
                    help='The number of threads for gRPC server.')
parser.add_argument('--prep-workers', default=12, type=int,
                    help='The number of threads for performing preprocessing.')



def serve():
    '''
    Initialize the data batch queue and start up the service.
    '''
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.grpc_workers))
    # server = grpc.server(futures.ProcessPoolExecutor(max_workers=32))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_all = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    data_transfer_pb2_grpc.add_DataTransferServicer_to_server(
                    DataTransferService(num_workers=args.prep_workers, transform_all=transform_all), 
                    server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on 'localhost:50051'.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()