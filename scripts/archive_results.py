import argparse
import shutil
import os
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--archive-dir", required=True)
    args = parser.parse_args()

    print(f"Archiving {args.results} to {args.archive_dir}")
    
    os.makedirs(args.archive_dir, exist_ok=True)
    
    src = Path(args.results)
    if not src.exists():
        print("Results file not found.")
        # Depending on workflow strictness, we might fail or just exit
        sys.exit(0)
        
    # Create a timestamped name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dst_name = f"benchmark_{timestamp}.json"
    dst = Path(args.archive_dir) / dst_name
    
    shutil.copy2(src, dst)
    print(f"Archived to {dst}")

if __name__ == "__main__":
    main()
