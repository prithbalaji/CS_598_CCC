#!/bin/bash
set -x
# Run both the train and storage servers locally and monitor their logs

# Ensure the virtual environment is created and activated
source scripts/setup_env.sh
source dl-processing-pipeline/training/dl-env/bin/activate

# Define cleanup function to stop background processes
cleanup() {
  echo "Stopping servers..."
  kill 0   # Kill all background processes spawned by this script
}

# Set trap to catch SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT

# Run the servers
echo "Starting storage_server.py..."
python dl-processing-pipeline/training/storage_server.py --batch_size 8 --compression 0 --offloading 0 &

echo "Starting train_server.py..."
python dl-processing-pipeline/training/train_server.py -a alexnet --gpu 0 --batch-size 8  &

# Wait for all background processes to finish
wait
