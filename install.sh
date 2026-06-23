#!/bin/bash

set -e

echo "Creating conda environment..."
conda create -n cellquantnet python=3.9 -y

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellquantnet

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Building depthwise_conv2d_implicit_gemm..."
cd CP-Net/model/cutlass/examples/19_large_depthwise_conv2d_torch_extension

rm -rf build *.egg-info dist

pip install -v . --no-build-isolation

cd ../../../../../../

echo "Installation completed successfully!"
