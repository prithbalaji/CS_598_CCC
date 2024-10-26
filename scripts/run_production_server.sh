#!/bin/bash
set -x
EXPORT PROD=1
source dl-env/bin/activate
python
echo "Production servers are running."