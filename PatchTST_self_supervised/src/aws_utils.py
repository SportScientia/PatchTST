import boto3
import pandas as pd
from io import BytesIO
import logging
import pickle
from typing import Any
import json
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3DataLoader:
    def __init__(self, bucket_name: str = None):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name

    def load_dataframe(self, file_name: str, bucket_name: str = None):
        """
        Load stock data from S3
        Returns DataFrame with columns: date, stock_code, price
        """
        try:
            bucket_name = bucket_name if bucket_name else self.bucket_name
            response = self.s3_client.get_object(
                Bucket=bucket_name,
                Key=file_name
            )
            if file_name.endswith('.csv.gz'):
                df = pd.read_csv(BytesIO(response['Body'].read()), compression='gzip')
            elif file_name.endswith('.csv'):
                df = pd.read_csv(BytesIO(response['Body'].read()))
            elif file_name.endswith('.parquet'):
                df = pd.read_parquet(BytesIO(response['Body'].read()))
            elif file_name.endswith('.pkl'):
                df = pd.read_pickle(BytesIO(response['Body'].read()))
            else:
                raise ValueError(f"Unsupported file type: {file_name}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
        

    def load_pickle(self, file_name: str, bucket_name: str = None):
        """
        Load pickle file from S3
        Returns object
        """
        try:
            bucket_name = bucket_name if bucket_name else self.bucket_name
            response = self.s3_client.get_object(
                Bucket=bucket_name,
                Key=file_name
            )
            return pickle.load(BytesIO(response['Body'].read()))
        except Exception as e:
            logger.error(f"Error loading pickle: {e}")
            return None
        
    def load_json(self, file_name: str, bucket_name: str = None):
        """
        Load json file from S3
        Returns object
        """
        try:
            bucket_name = bucket_name if bucket_name else self.bucket_name
            response = self.s3_client.get_object(
                Bucket=bucket_name,
                Key=file_name
            )
            return json.loads(response['Body'].read())
        except Exception as e:
            logger.error(f"Error loading json: {e}")
            return None
        
        

class S3Writer:
    def __init__(self, bucket_name: str = None):
        """Initialize S3Writer with S3 client"""
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name

    def save_pickle(self, file_name: str, obj: Any, bucket_name: str = None):
        """
        Save pickle file to S3
        Args:
            file_name: Name of the file to save
            obj: Object to save
            bucket_name: Name of the bucket to save to
        """
        try:
            bucket_name = bucket_name if bucket_name else self.bucket_name
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=file_name,
                Body=pickle.dumps(obj)
            )
        except Exception as e:
            logger.error(f"Error saving pickle: {e}")

    def save_json(self, file_name: str, obj: Any, bucket_name: str = None):
        """
        Save json file to S3
        Args:
            file_name: Name of the file to save
            obj: Object to save
            bucket_name: Name of the bucket to save to
        """
        try:
            bucket_name = bucket_name if bucket_name else self.bucket_name
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=file_name,
                Body=json.dumps(obj)
            )
        except Exception as e:
            logger.error(f"Error saving json: {e}")


def get_ephemeral_storage_path(instance_type: str = None):
    """Get ephemeral storage path for current EC2 instance"""
    try:
        # Get instance type
        instance_type = instance_type if instance_type else requests.get('http://169.254.169.254/latest/meta-data/instance-type').text
        
        # Common ephemeral storage paths by instance type
        ephemeral_paths = {
            't2.micro': '/dev/xvda',  # Usually no ephemeral storage
            't3.micro': '/dev/xvda',  # Usually no ephemeral storage
            'm5.large': '/dev/nvme1n1',
            'm5.xlarge': '/dev/nvme1n1',
            'c5.large': '/dev/nvme1n1',
            'c5.xlarge': '/dev/nvme1n1',
            'r5.large': '/dev/nvme1n1',
            'r5.xlarge': '/dev/nvme1n1',
            'g4dn.xlarge': '/opt/dlami/nvme',
            'g5.xlarge': '/opt/dlami/nvme'
            # Add more instance types as needed
        }
        
        if instance_type in ephemeral_paths:
            return ephemeral_paths[instance_type]
        else:
            # Fallback: check common locations
            return '/dev/nvme1n1'  # Most common for newer instance types
            
    except Exception as e:
        print(f"Error getting ephemeral storage path: {e}")
        return None