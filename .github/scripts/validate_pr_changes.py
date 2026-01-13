
import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


def validate_pr_changes(pr_files: List[str]) -> Dict:
    """
    Validate that PR only contains allowed changes.
    Prevents malicious PRs from modifying workflow files or scripts.
    """
    
    ALLOWED_PATHS = [
        'models/submissions/',
        'docs/',
        'README.md'
    ]
    
    FORBIDDEN_PATHS = [
        '.github/workflows/',
        '.github/scripts/',
        'benchmarks/evaluation/',
        'scripts/'
    ]
    
    issues = []
    warnings = []
    
    for file_path in pr_files:
        # Check if file is in allowed path
        is_allowed = any(file_path.startswith(path) for path in ALLOWED_PATHS)
        
        if not is_allowed:
            # Check if in forbidden path
            is_forbidden = any(file_path.startswith(path) for path in FORBIDDEN_PATHS)
            
            if is_forbidden:
                issues.append(
                    f"❌ File '{file_path}' modifies protected path. "
                    "PRs cannot modify workflow or evaluation code."
                )
            else:
                warnings.append(
                    f"⚠️  File '{file_path}' is outside standard submission paths. "
                    "Please only modify files in models/submissions/ or docs/"
                )
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'files_checked': len(pr_files)
    }
