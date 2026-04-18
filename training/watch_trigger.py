#!/usr/bin/env python3
"""
watch_trigger.py — Automated retraining trigger watcher.

Deployed as a K8S CronJob running every 30 minutes.
Polls data-proj01/triggers/ for new trigger files written by the
data team's batch_pipeline.py.

When a trigger is found:
  1. Reads the trigger JSON to get the dataset path and version
  2. Launches train.py with that dataset
  3. Deletes the trigger so it doesn't fire again

Trigger JSON format (written by data team):
{
  "trigger_version": "v20260420_001",
  "new_samples": 500,
  "dataset_path": "data-proj01/processed/train_v20260420.json",
  "created_at": "2026-04-20T10:00:00Z"
}

K8S CronJob schedule: */30 * * * *
K8S namespace: monitoring-proj01
Container image: subst-training:v{git_sha}
"""

import json
import os
import subprocess
import sys

import boto3

s3 = boto3.client('s3',
    endpoint_url=os.getenv('OS_ENDPOINT'),
    aws_access_key_id=os.getenv('OS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('OS_SECRET_KEY'))

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI',
                        'http://mlflow.monitoring-proj01:5000')

print('[watch_trigger] Checking data-proj01/triggers/ ...')

result = s3.list_objects_v2(Bucket='data-proj01', Prefix='triggers/')
triggers = result.get('Contents', [])

if not triggers:
    print('[watch_trigger] No triggers found. Exiting.')
    sys.exit(0)

for t in triggers:
    key = t['Key']
    # Skip directory-marker keys
    if key.endswith('/'):
        continue

    obj = s3.get_object(Bucket='data-proj01', Key=key)
    config = json.loads(obj['Body'].read())

    version = config.get('trigger_version', 'unknown')
    new_samples = config.get('new_samples', 0)
    dataset_path = config.get('dataset_path', '')

    print(f'[watch_trigger] Found trigger: {version} '
          f'({new_samples} new samples)')
    print(f'[watch_trigger] Dataset: {dataset_path}')

    # Launch training
    cmd = [
        'python', 'train.py',
        '--config', 'config.yaml',
        '--dataset', dataset_path,
        '--run_name', version,
        '--mlflow_tracking_uri', MLFLOW_URI,
    ]
    print(f'[watch_trigger] Running: {" ".join(cmd)}')

    try:
        subprocess.run(cmd, check=True)
        print(f'[watch_trigger] Training completed for {version}')
    except subprocess.CalledProcessError as e:
        print(f'[watch_trigger] Training FAILED for {version}: {e}')
        # Don't delete trigger on failure so it retries next cycle
        continue

    # Delete trigger after successful consumption
    s3.delete_object(Bucket='data-proj01', Key=key)
    print(f'[watch_trigger] Trigger consumed and deleted: {key}')

print('[watch_trigger] All triggers processed.')
