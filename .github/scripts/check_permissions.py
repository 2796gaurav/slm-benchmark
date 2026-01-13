#!/usr/bin/env python3
"""
SLM Benchmark - Security and Rate Limiting Scripts
Prevents abuse of GitHub Actions and GPU resources
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

def check_permissions(commenter: str, owner: str) -> Dict:
    """
    Check if commenter is authorized to trigger benchmarks.
    Only repository owner can trigger /test-benchmark and /push-results.
    """
    authorized = commenter.lower() == owner.lower()
    
    result = {
        'authorized': authorized,
        'commenter': commenter,
        'owner': owner,
        'timestamp': datetime.now().isoformat()
    }
    
    if not authorized:
        result['message'] = (
            f"@{commenter} is not authorized to trigger benchmarks. "
            f"Only @{owner} (repository owner) can run benchmarks."
        )
    
    return result


def main_check_permissions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commenter', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--output', default='authorized.json')
    args = parser.parse_args()
    
    result = check_permissions(args.commenter, args.owner)
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    if result['authorized']:
        print(f"✅ User @{args.commenter} is authorized")
    else:
        print(f"❌ User @{args.commenter} is NOT authorized")
        print(result['message'])
    
    sys.exit(0 if result['authorized'] else 1)