import multiprocessing as mp
from concurrent import futures
import grpc
import data_feed_pb2
import data_feed_pb2_grpc
import torch
from torchvision import transforms
import os
import zlib
from io import BytesIO
import argparse
from utils import DecodeJPEG, ConditionalNormalize, ImagePathDataset, load_logging_config
import logging
import psutil
import GPUtil
import time

from PIL import Image

kill = mp.Event()  # Global event to signal termination
num_cores = mp.cpu_count()

LOGGER = logging.getLogger()
DATA_LOGGER = logging.getLogger("data_collection")

if os.environ.get("PROD") is None:
    IMAGENET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet")
else:
    IMAGENET_PATH = "/workspace/data/imagenet"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Start the data feed server with an offloading plan."
    )
    parser.add_argument(
        "--offloading",
        type=int,
        default=0,
        help="Set t0 0 for no offloading, 1 for full offloading, or 2 for dynamic offloading.",
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=0,
        help="Set to 1 to enable compression before sending the sample.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=200, help="Batch size for loading images."
    )
    return parser.parse_args()


def handle_termination(signum, frame):
    LOGGER.warning("Termination signal received. Stopping workers...")
    kill.set()  # Set the event to stop the fill_queue process


class MetricsCollector:
    def __init__(self, interval=1):
        self.interval = interval
        self.running = False
        self.metrics = {
            'cpu_util': [],
            'memory_util': [],
            'disk_io_read': [],
            'disk_io_write': [],
            'network_sent': [],
            'network_recv': [],
        }
        if torch.cuda.is_available():
            self.metrics['gpu_util'] = []
            self.metrics['gpu_memory'] = []

    def start(self):
        self.running = True
        self.collection_thread = mp.Process(target=self._collect_metrics)
        self.collection_thread.start()

    def stop(self):
        self.running = False
        self.collection_thread.join()

    def _collect_metrics(self):
        last_disk_io = psutil.disk_io_counters()
        last_net_io = psutil.net_io_counters()
        while self.running:
            # CPU Utilization
            self.metrics['cpu_util'].append(psutil.cpu_percent())
            
            # Memory Utilization
            self.metrics['memory_util'].append(psutil.virtual_memory().percent)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            self.metrics['disk_io_read'].append(disk_io.read_bytes - last_disk_io.read_bytes)
            self.metrics['disk_io_write'].append(disk_io.write_bytes - last_disk_io.write_bytes)
            last_disk_io = disk_io

            # Network I/O
            net_io = psutil.net_io_counters()
            self.metrics['network_sent'].append(net_io.bytes_sent - last_net_io.bytes_sent)
            self.metrics['network_recv'].append(net_io.bytes_recv - last_net_io.bytes_recv)
            last_net_io = net_io

            # GPU metrics if available
            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.metrics['gpu_util'].append(gpus[0].load * 100)
                    self.metrics['gpu_memory'].append(gpus[0].memoryUtil * 100)

            DATA_LOGGER.info(f"Metrics: CPU: {self.metrics['cpu_util'][-1]}%, "
                             f"Memory: {self.metrics['memory_util'][-1]}%, "
                             f"Disk Read: {self.metrics['disk_io_read'][-1]}, "
                             f"Disk Write: {self.metrics['disk_io_write'][-1]}, "
                             f"Net Sent: {self.metrics['network_sent'][-1]}, "
                             f"Net Recv: {self.metrics['network_recv'][-1]}")
            
            if torch.cuda.is_available():
                DATA_LOGGER.info(f"GPU: {self.metrics['gpu_util'][-1]}%, "
                                 f"GPU Memory: {self.metrics['gpu_memory'][-1]}%")

            time.sleep(self.interval)


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
            LOGGER.info(
                f"Updated offloading plan: Sample {sample_id}, Transformations {transformations}"
            )

        # Respond with preprocessed samples
        while not kill.is_set():
            sample = self.q.get()  # Get the next sample from the queue
            yield data_feed_pb2.SampleBatch(
                samples=[
                    data_feed_pb2.Sample(
                        image=sample[0],
                        label=sample[1],
                        transformations_applied=sample[
                            2
                        ],  # Send the applied transformations count
                        is_compressed=sample[3],  # Send compression status
                    )
                ]
            )


def fill_queue(
    q,
    kill,
    batch_size,
    dataset_path,
    offloading_plan,
    offloading_value,
    compression_value,
    worker_id,
):
    # Custom decode transformation
    decode_jpeg = DecodeJPEG()

    transformations = [
        decode_jpeg,  # Decode raw JPEG bytes to a PIL image
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Converts PIL images to tensors
        ConditionalNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # Ensure that ImageFolder uses the transform to convert images to tensors
    dataset = ImagePathDataset(os.path.join(dataset_path, "train"))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    for batch_idx, (data, target) in enumerate(loader):
        LOGGER.debug(
            f"Worker {worker_id} - Batch {batch_idx}: Loaded {len(data)} images."
        )
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
                transformed_data.save(img_byte_arr, format="JPEG")
                transformed_data = img_byte_arr.getvalue()  # Get image in bytes
            elif isinstance(transformed_data, torch.Tensor):
                # If it's a PyTorch tensor, convert to numpy and then to bytes
                transformed_data = (
                    transformed_data.numpy().tobytes()
                )  # Convert tensor to numpy and Serialize numpy array to bytes
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
                    q.put(
                        (
                            transformed_data,
                            target[i],
                            num_transformations,
                            is_compressed,
                        ),
                        timeout=1,
                    )
                    added = True
                except:
                    continue


def serve(offloading_value, compression_value, batch_size):
    q = mp.Queue(maxsize=2 * num_cores)

    # Cache for storing the offloading plan (sample_id -> number of transformations)
    offloading_plan = {}

    metrics_collector = MetricsCollector()
    metrics_collector.start()

    # Start the fill_queue process
    workers = []
    for worker_id in range(num_cores):
        p = mp.Process(
            target=fill_queue,
            args=(
                q,
                kill,
                batch_size,
                IMAGENET_PATH,
                offloading_plan,
                offloading_value,
                compression_value,
                worker_id,
            ),
        )
        workers.append(p)
        p.start()

    # Start the gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        # Client and Server: Increase message sizes and control flow limits
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),  # 1 GB
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),  # 1 GB
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 10000),
        ],
    )
    data_feed_pb2_grpc.add_DataFeedServicer_to_server(
        DataFeedService(q, offloading_plan), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        LOGGER.info("Keyboard interrupt received. Stopping server...")
    finally:
        kill.set()
        for p in workers:
            p.join()
        metrics_collector.stop()
        LOGGER.info("Server stopped.")
    


def custom_collate_fn(batch):
    raw_images = []
    targets = []
    for img_path, target in batch:
        if os.path.isfile(img_path):
            with open(img_path, "rb") as f:
                raw_img_data = f.read()  # Read the raw JPEG image in binary
            raw_images.append(raw_img_data)
            targets.append(target)

    return raw_images, targets  # Return two lists: images and targets


if __name__ == "__main__":
    load_logging_config()
    args = parse_args()

    # Example usage of the --offloading argument
    offloading_value = args.offloading
    compression_value = args.compression
    batch_size = args.batch_size
    serve(offloading_value, compression_value, batch_size)

