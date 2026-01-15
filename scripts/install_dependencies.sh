#!/bin/bash
# Installation script that handles the codecarbon/bfcl-eval dependency conflict
# by installing codecarbon without fief-client (which is only needed for CLI)

set -e

echo "Installing base dependencies..."
pip install -r requirements.txt

echo "Installing codecarbon dependencies (without fief-client)..."
# Install codecarbon's dependencies manually, excluding fief-client
pip install arrow click "pandas>=2.3.3; python_version >= '3.14'" "pandas; python_version < '3.14'" prometheus_client "psutil>=6.0.0" py-cpuinfo pydantic nvidia-ml-py rapidfuzz requests questionary rich typer

echo "Installing codecarbon without dependencies..."
pip install --no-deps codecarbon==3.2.1

echo "Verifying installation..."
python -c "from codecarbon import EmissionsTracker; print('✓ codecarbon imported successfully')"
python -c "import bfcl_eval; print('✓ bfcl-eval imported successfully')"

echo "✓ All dependencies installed successfully!"
