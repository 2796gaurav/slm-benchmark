#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Local CPU Benchmark Test...${NC}"

# Create a temporary test submission
cat <<EOF > models/submissions/test_submission.yaml
model:
  name: "TinyTestModel"
  family: "Test"
  version: "1.0"
  hf_repo: "HuggingFaceTB/SmolLM2-135M"
  parameters: "135M"
  architecture: "llama"
  context_length: 2048
  license: "Apache-2.0"
  quantizations:
    - name: "FP32" # Use FP32 for CPU safety if bf16 not supported
      format: "transformers"
  categories:
    - "test"
  testing:
    priority: "fast"
EOF

# Ensure results directory exists
mkdir -p results/raw

# Run the benchmark script directly
# We set CUDA_VISIBLE_DEVICES="" to force CPU visibility checks
export CUDA_VISIBLE_DEVICES=""

echo "Running benchmark script..."
python benchmarks/evaluation/run_benchmark.py \
    --submission-file models/submissions/test_submission.yaml \
    --output-dir results/raw/ \
    --save-logs \
    --deterministic \
    --seed 42 \
    --timestamp $(date +%Y%m%d_%H%M%S) \
    --limit 5

echo -e "${GREEN}Test completed! Results saved to results/raw/TinyTestModel/${NC}"
