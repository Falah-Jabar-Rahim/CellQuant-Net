#!/bin/bash

set -e

ENV_NAME="cellquantnet"
CUTLASS_ZIP_URL="https://github.com/DingXiaoH/RepLKNet-pytorch/raw/main/cutlass.zip"
CUTLASS_DIR="CP-Net/model/cutlass"

echo "Creating conda environment..."
conda create -n "$ENV_NAME" python=3.9 -y

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading CUTLASS..."
mkdir -p CP-Net/model

if [ ! -d "$CUTLASS_DIR" ]; then
    wget -O cutlass.zip "$CUTLASS_ZIP_URL"
    unzip cutlass.zip -d CP-Net/model
    mv CP-Net/model/cutlass-master "$CUTLASS_DIR" 2>/dev/null || true
    rm cutlass.zip
fi

echo "Building depthwise_conv2d_implicit_gemm..."
cd "$CUTLASS_DIR/examples/19_large_depthwise_conv2d_torch_extension"

rm -rf build dist *.egg-info

pip install -v . --no-build-isolation

cd -

echo "Installation completed successfully!"
