import json, os, subprocess, sys, boto3

s3 = boto3.client('s3',
    endpoint_url=os.getenv('OS_ENDPOINT'),
    aws_access_key_id=os.getenv('OS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('OS_SECRET_KEY'))

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow.monitoring-proj01:5000')
result   = s3.list_objects_v2(Bucket='data-proj01', Prefix='triggers/')
triggers = result.get('Contents', [])

if not triggers:
    print('[watch_trigger] No triggers found. Exiting.')
    sys.exit(0)

for t in triggers:
    key = t['Key']
    if key.endswith('/'): continue
    cfg = json.loads(s3.get_object(Bucket='data-proj01', Key=key)['Body'].read())
    print(f'[watch_trigger] {cfg["trigger_version"]} ({cfg["new_samples"]} samples)')
    try:
        subprocess.run([
            'python', 'train.py',
            '--dataset',  cfg['dataset_path'],
            '--run_name', cfg['trigger_version'],
            '--mlflow_tracking_uri', MLFLOW_URI
        ], check=True)
        s3.delete_object(Bucket='data-proj01', Key=key)
        print(f'[watch_trigger] Consumed and deleted: {key}')
    except subprocess.CalledProcessError as e:
        print(f'[watch_trigger] FAILED (not deleting trigger): {e}')
