#!/usr/bin/env python3
"""Sync registry to website data"""

import json
import shutil
from pathlib import Path

def update_website(registry_file: str, output_file: str):
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