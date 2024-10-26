from PIL import Image
from io import BytesIO
from torchvision import transforms
import torch
import grpc
import data_feed_pb2_grpc
import numpy as np
import zlib
import torch.utils.data
import os
from torch.utils.data import Dataset
from logging.config import dictConfig
import json

class DecodeJPEG:
    def __call__(self, raw_bytes):
        return Image.open(BytesIO(raw_bytes))


class ConditionalNormalize:
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, tensor):
        # Only apply normalization if the tensor has 3 channels
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(
                3, 1, 1
            )  # Repeat the single channel across the 3 RGB channels

        # Apply normalization to 3-channel (RGB) images
        return self.normalize(tensor)


class RemoteDataset(torch.utils.data.IterableDataset):
    def __init__(self, host, port, batch_size=256):
        self.host = host
        self.port = port
        self.batch_size = batch_size

    def __iter__(self):
        channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 1024 * 1024 * 1024),  # 1 GB
                ("grpc.max_receive_message_length", 1024 * 1024 * 1024),  # 1 GB
                ("grpc.http2.max_pings_without_data", 0),  # No limit
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.min_ping_interval_without_data_ms", 10000),
            ],
        )

        stub = data_feed_pb2_grpc.DataFeedStub(channel)

        samples = stub.StreamSamples(iter([]))
        batch_images = []
        batch_labels = []

        for i, sample_batch in enumerate(samples):
            for s in sample_batch.samples:
                if s.is_compressed:
                    # Decompress the image data
                    decompressed_image = zlib.decompress(s.image)
                else:
                    decompressed_image = (
                        s.image
                    )  # No need to decompress if it's not compressed
                if s.transformations_applied < 5:
                    processed_image, _, _ = self.preprocess_sample(
                        decompressed_image, s.transformations_applied
                    )
                else:
                    img_np = np.frombuffer(
                        decompressed_image, dtype=np.float32
                    )  # Adjust dtype if necessary
                    img_np = img_np.reshape(
                        (3, 224, 224)
                    )  # Reshape based on original image dimensions
                    processed_image = torch.tensor(
                        img_np
                    )  # Convert NumPy array to PyTorch tensor
                # Convert label to tensor
                label = torch.tensor(s.label)  # Directly convert the label to a tensor

                batch_images.append(processed_image)
                batch_labels.append(label)

                # Yield the batch when it reaches the batch_size
                if len(batch_images) == self.batch_size:
                    yield torch.stack(batch_images), torch.stack(batch_labels)
                    batch_images = []
                    batch_labels = []
        # Yield any remaining samples as the final batch
        if batch_images:
            yield torch.stack(batch_images), torch.stack(batch_labels)

    def preprocess_sample(self, sample, transformations_applied):
        # List of transformations to apply individually
        decode_jpeg = DecodeJPEG()

        transformations = [
            decode_jpeg,  # Decode raw JPEG bytes to a PIL image
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Converts PIL images to tensors
            ConditionalNormalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Conditional normalization
        ]

        processed_sample = sample
        for i in range(transformations_applied, len(transformations)):
            if transformations[i] is not None:
                processed_sample = transformations[i](processed_sample)
        return processed_sample, None, None


class ImagePathDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.targets = []

        for dirpath, _, filenames in os.walk(root_dir):
            class_name = os.path.basename(dirpath)  # Use directory name as the class label
            class_idx = self.get_class_index(class_name)  # Define or map class indices as needed

            for filename in filenames:
                img_path = os.path.join(dirpath, filename)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.targets.append(class_idx)

    def get_class_index(self, class_name):
        # This function can map class names to indices (or use a dictionary as needed)
        # For simplicity, it’s using a hash-based mapping here:
        return hash(class_name) % 1000  # Or use a custom mapping logic

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        target = self.targets[idx]

        return img_path, target  # Only return two values: path and target


def load_logging_config():
    """Configure logging handlers"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{script_dir}/logging.json", 'r') as read_file:
        logging_config = json.load(read_file)

    if os.environ.get("PROD") is None:
        data_collection_log_file_local_ = os.path.join(script_dir, "logs", "data_collection.log")
        debug_log_file_production = os.path.join(script_dir, "logs", "debug_log.log")
        logging_config['handlers']['data_collection_handler']['filename'] = data_collection_log_file_local_
        logging_config['handlers']['file']['filename'] = debug_log_file_production

    dictConfig(logging_config)
