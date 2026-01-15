#!/usr/bin/env python3
"""Sync registry to website data"""

import json
import shutil
from pathlib import Path

def update_website(registry_file: str, output_file: str):
    """Copy registry to website data directory"""
    import os
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Verify registry file exists
    if not os.path.exists(registry_file):
        raise FileNotFoundError(f"Registry file not found: {registry_file}")
    
    # Copy registry to website data
    shutil.copy(registry_file, output_file)
    print(f"✅ Website data updated: {output_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    update_website(args.registry, args.output)