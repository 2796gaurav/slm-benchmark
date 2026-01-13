import os
import modal
import subprocess
import sys
from pathlib import Path

# Define the image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "scipy",
        "numpy",
        "pandas",
        "tqdm",
        "pyyaml",
        "huggingface_hub"
    )
    .apt_install("git")
)

app = modal.App("slm-benchmark-runner", image=image)

# Mount the codebase
# We mount the entire current directory to /root/workspace
benchmark_mount = modal.Mount.from_local_dir(
    ".",
    remote_path="/root/workspace",
    condition=lambda p: not any(x in p for x in [".git", "__pycache__", "venv", ".venv", "results"])
)

@app.function(
    gpu="T4",
    timeout=3600,  # 1 hour timeout
    mounts=[benchmark_mount],
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # Assumes user has this secret in Modal
        # Or we can pass it via env if it comes from GH Actions secrets
    ],
    cpu=4.0,
    memory=16384,
)
def run_benchmark_on_modal(submission_file: str, output_dir: str, timestamp: str):
    """
    Runs the benchmark script on a Modal T4 instance.
    """
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting benchmark for {submission_file}")
    
    # Change to workspace
    os.chdir("/root/workspace")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct command
    cmd = [
        "python",
        "benchmarks/evaluation/run_benchmark.py",
        "--submission-file", submission_file,
        "--output-dir", output_dir,
        "--save-logs",
        "--deterministic",
        "--seed", "42",
        "--timestamp", timestamp,
        "--full-report"
    ]
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    # Run the benchmark
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        
        # Read the output files and return them so we can save them locally
        # This is important because the local container file system is ephemeral
        output_data = {}
        out_path = Path(output_dir)
        if out_path.exists():
            for file_path in out_path.glob("**/*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(out_path)
                    with open(file_path, "rb") as f:
                        output_data[str(rel_path)] = f.read()
        
        return {"status": "success", "output": output_data}

    except subprocess.CalledProcessError as e:
        logger.error("Benchmark failed!")
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        return {"status": "error", "error": str(e), "stdout": e.stdout, "stderr": e.stderr}

@app.local_entrypoint()
def main(submission_file: str = "models/submissions/*.yaml", output_dir: str = "results/raw/"):
    import glob
    import time
    
    # Expand wildcard for local testing/submission finds
    # In GitHub Actions, we might pass a specific file
    files = glob.glob(submission_file)
    if not files:
        print(f"No submission files found matching {submission_file}")
        return

    target_file = files[0] # Just take the first one for now
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print(f"Sending job to Modal for {target_file}...")
    
    ret = run_benchmark_on_modal.remote(target_file, output_dir, timestamp)
    
    if ret["status"] == "success":
        print("Job completed successfully. Saving results...")
        out_base = Path(output_dir)
        out_base.mkdir(parents=True, exist_ok=True)
        
        for rel_path, data in ret["output"].items():
            final_path = out_base / rel_path
            final_path.parent.mkdir(parents=True, exist_ok=True)
            with open(final_path, "wb") as f:
                f.write(data)
        print(f"Results saved to {out_base}")
    else:
        print("Job failed!")
        sys.exit(1)
