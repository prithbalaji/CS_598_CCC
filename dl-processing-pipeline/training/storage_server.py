import multiprocessing as mp
from concurrent import futures
import grpc
import data_feed_pb2
import data_feed_pb2_grpc
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
import os
import zlib
import time
from io import BytesIO
import argparse
from utils import DecodeJPEG, ConditionalNormalize, ImagePathDataset
import logging
import json
from logging.config import dictConfig
<<<<<<< HEAD
=======

LOGGER = logging.getLogger()

def load_logging_config():
    with open('logging.json') as read_file:
        dictConfig(json.load(read_file))

>>>>>>> f7d7822 (Modify dataset path)

from PIL import Image
kill = mp.Event()  # Global event to signal termination
num_cores = mp.cpu_count()

LOGGER = logging.getLogger()
DATA_LOGGER = logging.getLogger('data_collection')

def load_logging_config():
    with open('logging.json') as read_file:
        dictConfig(json.load(read_file))


def parse_args():
    parser = argparse.ArgumentParser(description="Start the data feed server with an offloading plan.")
    parser.add_argument('--offloading', type=int, default=0, help='Set t0 0 for no offloading, 1 for full offloading, or 2 for dynamic offloading.')
    parser.add_argument('--compression', type=int, default=0, help='Set to 1 to enable compression before sending the sample.')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for loading images.')
    return parser.parse_args()

def handle_termination(signum, frame):
    LOGGER.warning("Termination signal received. Stopping workers...")
    kill.set()  # Set the event to stop the fill_queue process

class DataFeedService(data_feed_pb2_grpc.DataFeedServicer):
    def __init__(self, q, offloading_plan):
        self.q = q
        self.offloading_plan = offloading_plan  # Store offloading plan for each sample

    def StreamSamples(self, request_iterator, context):
        # Listen for updates to the offloading plan
        for request in request_iterator:
            sample_id = request.sample_id
            transformations = request.transformations
            self.offloading_plan[sample_id] = transformations
            LOGGER.info(f"Updated offloading plan: Sample {sample_id}, Transformations {transformations}")

        # Respond with preprocessed samples
        while not kill.is_set():
            sample = self.q.get()  # Get the next sample from the queue
            yield data_feed_pb2.SampleBatch(
                samples=[data_feed_pb2.Sample(
                    image=sample[0],
                    label=sample[1],
                    transformations_applied=sample[2],  # Send the applied transformations count
                    is_compressed=sample[3]  # Send compression status
                )]
            )

def fill_queue(q, kill, batch_size, dataset_path, offloading_plan, offloading_value, compression_value, worker_id):
    # Custom decode transformation
    decode_jpeg = DecodeJPEG()

    transformations = [
        decode_jpeg,  # Decode raw JPEG bytes to a PIL image
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Converts PIL images to tensors
        ConditionalNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # Ensure that ImageFolder uses the transform to convert images to tensors
    dataset = ImagePathDataset(os.path.join(dataset_path, 'train'))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=16, pin_memory=True, collate_fn=custom_collate_fn)

    for batch_idx, (data, target) in enumerate(loader):
        LOGGER.debug(f"Worker {worker_id} - Batch {batch_idx}: Loaded {len(data)} images.")
        for i in range(len(data)):  # Loop over individual samples
            sample_id = batch_idx * batch_size + i
            if offloading_value == 0:
                num_transformations = 0
            elif offloading_value == 1:
                num_transformations = 5
            else:
                num_transformations = offloading_plan.get(sample_id, 0)

            transformed_data = data[i]
            for j in range(min(num_transformations, 5)):
                transformed_data = transformations[j](transformed_data)

            # Serialize the transformed data
            if isinstance(transformed_data, Image.Image):
                # If it's still a PIL image, convert it to bytes
                img_byte_arr = BytesIO()
                transformed_data.save(img_byte_arr, format='JPEG')
                transformed_data = img_byte_arr.getvalue()  # Get image in bytes
            elif isinstance(transformed_data, torch.Tensor):
                # If it's a PyTorch tensor, convert to numpy and then to bytes
                transformed_data = transformed_data.numpy().tobytes()  # Convert tensor to numpy and Serialize numpy array to bytes
            if compression_value == 1:
                transformed_data = zlib.compress(transformed_data)  # Compress data
                is_compressed = True
            else:
                is_compressed = False
            # time.sleep(1) this was used to simulare low network bandwidth but it is a crude proxy

            # Add the sample and the number of applied transformations to the queue
            added = False
            while not added and not kill.is_set():
                try:
                    q.put((transformed_data, target[i], num_transformations, is_compressed), timeout=1)
                    added = True
                except:
                    continue




def serve(offloading_value, compression_value, batch_size):
    q = mp.Queue(maxsize=2 * num_cores)

    # Cache for storing the offloading plan (sample_id -> number of transformations)
    offloading_plan = {}

    # Start the fill_queue process
    workers = []
    for worker_id in range(num_cores):
        p = mp.Process(target=fill_queue, args=(q, kill, batch_size, '/data/imagenet', offloading_plan, offloading_value, compression_value, worker_id))
        workers.append(p)
        p.start()

    # Start the gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        # Client and Server: Increase message sizes and control flow limits
        options = [
                ('grpc.max_send_message_length', 1024 * 1024 * 1024),  # 1 GB
                ('grpc.max_receive_message_length', 1024 * 1024 * 1024),  # 1 GB
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 10000)
            ]

    )
    data_feed_pb2_grpc.add_DataFeedServicer_to_server(DataFeedService(q, offloading_plan), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

    kill.set()
    for p in workers:
        p.join()

def custom_collate_fn(batch):
    raw_images = []
    targets = []
    for img_path, target in batch:
        with open(img_path, 'rb') as f:
            raw_img_data = f.read()  # Read the raw JPEG image in binary
        raw_images.append(raw_img_data)
        targets.append(target)

    return raw_images, targets  # Return two lists: images and targets


if __name__ == '__main__':
    args = parse_args()

    # Example usage of the --offloading argument
    offloading_value = args.offloading
    compression_value = args.compression
    batch_size = args.batch_size
    serve(offloading_value, compression_value, batch_size)
