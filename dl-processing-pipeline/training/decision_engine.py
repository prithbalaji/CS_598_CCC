import grpc
import data_feed_pb2
import data_feed_pb2_grpc
import logging
from utils import load_logging_config

LOGGER = logging.getLogger()


class DecisionEngine:
    def __init__(
        self,
        sample_metrics,
        gpu_t,
        cpu_t,
        io_t,
        cpu_cores_compute,
        cpu_cores_storage,
        grpc_host,
        grpc_port,
    ):
        """
        :param sample_metrics: List of tuples (original_size, transformed_size, preprocessing_time)
        :param tg: GPU time for one epoch (TG)
        :param tcc: CPU time on compute node (TCC)
        :param tcs: CPU time on storage node (TCS)
        :param tnet: Network transfer time (TNet)
        :param cpu_cores_compute: Number of CPU cores on compute node
        :param cpu_cores_storage: Number of CPU cores on storage node
        :param grpc_host: Hostname for the gRPC server
        :param grpc_port: Port for the gRPC server
        """
        self.sample_metrics = sample_metrics
        self.num_samples = len(sample_metrics)
        self.tg = self.num_samples / gpu_t  # Total GPU time for all samples
        self.tcc = self.num_samples / cpu_t
        self.tcs = self.num_samples / cpu_t
        self.tnet = self.num_samples / io_t
        self.cpu_cores_compute = cpu_cores_compute
        self.cpu_cores_storage = cpu_cores_storage
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.offloading_plan = {}

    def decide_offloading(self, current_offloading_plan):
        decisions = []
        LOGGER.info(f"Sample_metrics data type = {type(self.sample_metrics)}")

        # Iterate through each sample in sample_metrics
        for sample_id, sample in enumerate(self.sample_metrics):
            original_size = sample["original_size"]
            transformed_sizes = sample["transformed_sizes"]
            preprocessing_times = sample["preprocessing_times"]
            current_transformations = current_offloading_plan.get(sample_id, 0)

            LOGGER.debug(
                f"Processing sample {sample_id}: original_size={original_size}, current_transformations={current_transformations}"
            )

            for i in range(current_transformations, len(preprocessing_times)):
                if i == 0:
                    size_reduction = original_size - transformed_sizes[i]
                else:
                    size_reduction = transformed_sizes[i - 1] - transformed_sizes[i]

                marginal_saved_tnet = (size_reduction / original_size) * self.tnet
                marginal_efficiency = marginal_saved_tnet / preprocessing_times[i]

                # Debugging output for decision process
                LOGGER.debug(
                    f"Sample {sample_id}, transformation {i}: size_reduction={size_reduction}, marginal_saved_tnet={marginal_saved_tnet}, "
                    f"marginal_efficiency={marginal_efficiency}, preprocessing_time={preprocessing_times[i]}"
                )

                if marginal_efficiency > 0:
                    decisions.append(
                        (
                            sample_id,
                            i + 1,
                            marginal_efficiency,
                            marginal_saved_tnet,
                            preprocessing_times[i],
                        )
                    )
                    LOGGER.debug(
                        f"Sample {sample_id}: Added to offloading plan with transformation {i+1} and marginal_efficiency={marginal_efficiency}"
                    )
                else:
                    LOGGER.debug(
                        f"Sample {sample_id}: Not selected for offloading because marginal_efficiency={marginal_efficiency}"
                    )

        return sorted(decisions, key=lambda x: -x[2])

    def iterative_offloading(self):
        current_tnet = self.tnet
        current_tcc = self.tcc / self.cpu_cores_compute
        current_tcs = 0
        current_offloading_plan = {}

        offloading_plan = {}
        decisions = self.decide_offloading(current_offloading_plan)

        for (
            sample_id,
            next_transformation,
            marginal_efficiency,
            marginal_saved_tnet,
            marginal_preprocessing_time,
        ) in decisions:
            current_tnet -= marginal_saved_tnet
            current_tcc -= marginal_preprocessing_time / self.cpu_cores_compute
            current_tcs += marginal_preprocessing_time / self.cpu_cores_storage

            LOGGER.debug(
                f"Sample {sample_id} selected for offloading: next_transformation={next_transformation}, "
                f"marginal_efficiency={marginal_efficiency}, current_tnet={current_tnet}, current_tcs={current_tcs}"
            )

            current_offloading_plan[sample_id] = next_transformation
            offloading_plan[sample_id] = next_transformation

            # Check if offloading should stop based on the TNet and TCS comparison
            if current_tnet < current_tcs:
                LOGGER.debug(
                    f"Stopping offloading: current_tnet={current_tnet} is less than current_tcs={current_tcs}"
                )
                break

        return offloading_plan

    def send_offloading_requests(self):
        # Create a gRPC channel to communicate with the storage server
        channel = grpc.insecure_channel(f"{self.grpc_host}:{self.grpc_port}")
        stub = data_feed_pb2_grpc.DataFeedStub(channel)

        # Create a stream for offloading requests
        offloading_stream = stub.StreamSamples(
            iter(self._generate_offloading_requests())
        )

        # Iterate through the stream to make sure the server acknowledges the updates
        for response in offloading_stream:
            LOGGER.debug(f"Server Response: {response.status}")

    def _generate_offloading_requests(self):
        # Create offloading plan and generate gRPC messages for the server
        offloading_plan = self.iterative_offloading()
        for sample_id, transformations in offloading_plan.items():
            yield data_feed_pb2.OffloadingRequest(
                sample_id=sample_id, transformations=transformations
            )


if __name__ == "__main__":
    load_logging_config()
    # Example input from profiling and broader system metrics
    sample_metrics = [
        (10000, 2000, 1.5),
        (9000, 3000, 0.5),
        (8000, 4000, 0.7),
    ]  # Example data
    tg = 10  # Example GPU time for one epoch
    tcc = 5  # Example CPU time for local preprocessing (compute node)
    tcs = 0  # Initial CPU time on storage node (no offloading)
    tnet = 8  # Example network transfer time (TNet)
    cpu_cores_compute = 8  # Example number of CPU cores on compute node
    cpu_cores_storage = 4  # Example number of CPU cores on storage node

    # Initialize the DecisionEngine with gRPC details
    engine = DecisionEngine(
        sample_metrics,
        tg,
        tcc,
        tcs,
        tnet,
        cpu_cores_compute,
        cpu_cores_storage,
        grpc_host="localhost",
        grpc_port=50051,
    )

    # Send offloading requests
    engine.send_offloading_requests()
