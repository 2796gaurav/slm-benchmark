
import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


class RateLimiter:
    """
    Rate limiting to prevent abuse of GitHub Actions and GPU resources.
    
    Limits:
    - Max 3 submissions per user per day
    - Max 1 benchmark run per model per day
    - Max 10 total benchmark runs per day across all users
    """
    
    RATE_LIMIT_FILE = '.github/rate_limits.json'
    MAX_SUBMISSIONS_PER_USER_PER_DAY = 3
    MAX_BENCHMARKS_PER_MODEL_PER_DAY = 1
    MAX_TOTAL_BENCHMARKS_PER_DAY = 10
    
    def __init__(self):
        self.load_state()
    
    def load_state(self):
        """Load rate limit state from file"""
        try:
            if os.path.exists(self.RATE_LIMIT_FILE):
                with open(self.RATE_LIMIT_FILE, 'r') as f:
                    self.state = json.load(f)
            else:
                self.state = {
                    'submissions': {},  # {user: [{date, model}, ...]}
                    'benchmarks': {},   # {model: [{date, status}, ...]}
                    'daily_total': {}   # {date: count}
                }
        except Exception as e:
            print(f"Warning: Could not load rate limit state: {e}")
            self.state = {'submissions': {}, 'benchmarks': {}, 'daily_total': {}}
    
    def save_state(self):
        """Save rate limit state to file"""
        try:
            os.makedirs(os.path.dirname(self.RATE_LIMIT_FILE), exist_ok=True)
            with open(self.RATE_LIMIT_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save rate limit state: {e}")
    
    def check_submission_rate(self, user: str) -> Dict:
        """Check if user has exceeded submission rate limit"""
        today = datetime.now().date().isoformat()
        
        if user not in self.state['submissions']:
            self.state['submissions'][user] = []
        
        # Filter to today's submissions
        today_submissions = [
            s for s in self.state['submissions'][user]
            if s['date'] == today
        ]
        
        allowed = len(today_submissions) < self.MAX_SUBMISSIONS_PER_USER_PER_DAY
        
        return {
            'allowed': allowed,
            'user': user,
            'count_today': len(today_submissions),
            'limit': self.MAX_SUBMISSIONS_PER_USER_PER_DAY,
            'resets_at': self._get_reset_time()
        }
    
    def check_benchmark_rate(self, model_name: str) -> Dict:
        """Check if model has exceeded benchmark rate limit"""
        today = datetime.now().date().isoformat()
        
        if model_name not in self.state['benchmarks']:
            self.state['benchmarks'][model_name] = []
        
        # Filter to today's benchmarks
        today_benchmarks = [
            b for b in self.state['benchmarks'][model_name]
            if b['date'] == today
        ]
        
        allowed = len(today_benchmarks) < self.MAX_BENCHMARKS_PER_MODEL_PER_DAY
        
        return {
            'allowed': allowed,
            'model': model_name,
            'count_today': len(today_benchmarks),
            'limit': self.MAX_BENCHMARKS_PER_MODEL_PER_DAY,
            'resets_at': self._get_reset_time()
        }
    
    def check_daily_total(self) -> Dict:
        """Check total benchmark runs today across all models"""
        today = datetime.now().date().isoformat()
        
        count_today = self.state['daily_total'].get(today, 0)
        allowed = count_today < self.MAX_TOTAL_BENCHMARKS_PER_DAY
        
        return {
            'allowed': allowed,
            'count_today': count_today,
            'limit': self.MAX_TOTAL_BENCHMARKS_PER_DAY,
            'resets_at': self._get_reset_time()
        }
    
    def record_submission(self, user: str, model_name: str):
        """Record a new submission"""
        today = datetime.now().date().isoformat()
        
        if user not in self.state['submissions']:
            self.state['submissions'][user] = []
        
        self.state['submissions'][user].append({
            'date': today,
            'model': model_name,
            'timestamp': datetime.now().isoformat()
        })
        
        self.save_state()
    
    def record_benchmark(self, model_name: str):
        """Record a new benchmark run"""
        today = datetime.now().date().isoformat()
        
        if model_name not in self.state['benchmarks']:
            self.state['benchmarks'][model_name] = []
        
        self.state['benchmarks'][model_name].append({
            'date': today,
            'timestamp': datetime.now().isoformat(),
            'status': 'started'
        })
        
        # Update daily total
        self.state['daily_total'][today] = self.state['daily_total'].get(today, 0) + 1
        
        self.save_state()
    
    def _get_reset_time(self) -> str:
        """Get time when rate limits reset (midnight UTC)"""
        tomorrow = datetime.now().date() + timedelta(days=1)
        reset_time = datetime.combine(tomorrow, datetime.min.time())
        return reset_time.isoformat()
    
    def cleanup_old_records(self, days_to_keep: int = 7):
        """Clean up old records to prevent file from growing indefinitely"""
        cutoff_date = (datetime.now().date() - timedelta(days=days_to_keep)).isoformat()
        
        # Clean submissions
        for user in self.state['submissions']:
            self.state['submissions'][user] = [
                s for s in self.state['submissions'][user]
                if s['date'] >= cutoff_date
            ]
        
        # Clean benchmarks
        for model in self.state['benchmarks']:
            self.state['benchmarks'][model] = [
                b for b in self.state['benchmarks'][model]
                if b['date'] >= cutoff_date
            ]
        
        # Clean daily totals
        self.state['daily_total'] = {
            date: count for date, count in self.state['daily_total'].items()
            if date >= cutoff_date
        }
        
        self.save_state()


def main_rate_limit():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submitter', required=True)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--check-frequency', action='store_true')
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    
    limiter = RateLimiter()
    
    # Clean up old records
    if args.check_frequency:
        limiter.cleanup_old_records()
    
    # Check submission rate
    submission_check = limiter.check_submission_rate(args.submitter)
    
    if not submission_check['allowed']:
        print(f"❌ Rate limit exceeded for user @{args.submitter}")
        print(f"   Submitted {submission_check['count_today']}/{submission_check['limit']} times today")
        print(f"   Limit resets at: {submission_check['resets_at']}")
        sys.exit(1)
    
    # Check model benchmark rate
    if args.model_name:
        benchmark_check = limiter.check_benchmark_rate(args.model_name)
        
        if not benchmark_check['allowed']:
            print(f"❌ Rate limit exceeded for model '{args.model_name}'")
            print(f"   Benchmarked {benchmark_check['count_today']}/{benchmark_check['limit']} times today")
            sys.exit(1)
        
        # Check daily total
        total_check = limiter.check_daily_total()
        
        if not total_check['allowed']:
            print(f"❌ Daily benchmark limit reached")
            print(f"   {total_check['count_today']}/{total_check['limit']} benchmarks run today")
            print(f"   This prevents excessive GPU usage")
            sys.exit(1)
        
        # Record if requested
        if args.record:
            limiter.record_submission(args.submitter, args.model_name)
            limiter.record_benchmark(args.model_name)
            print(f"✅ Recorded submission and benchmark for {args.model_name}")
    
    print(f"✅ Rate limits OK for @{args.submitter}")
    sys.exit(0)