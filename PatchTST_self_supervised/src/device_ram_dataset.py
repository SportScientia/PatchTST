import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from src.aws_utils import S3DataLoader, get_ephemeral_storage_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeviceRAMForceDataset(Dataset):
    """Dataset that loads all batches into RAM and optionally transfers them to device upfront"""
    
    def __init__(self, batches_dir: str, fold: int, batch_size: int, subjects: Optional[List[str]] = None, device: Optional[torch.device] = None, bucket: str = None):
        """
        Initialize the device RAM dataset
        
        Args:
            batches_dir: Directory containing the prepared batches
            subjects: List of subject IDs to include. If None, includes all subjects.
            device: Device to transfer data to. If None, data stays on CPU.
        """
        self.bucket = bucket
        self.batches_dir = Path(batches_dir)
        self.device = device
        self.batch_size = batch_size

        if bucket:
            loader = S3DataLoader(bucket_name=bucket)
            #### check if the data exists in the ephemeral storage, load from there if it does
            ephemeral_storage_path = get_ephemeral_storage_path()
            logger.info(f'Checking for cached data at {ephemeral_storage_path}')
            ephemeral_storage_path = Path(ephemeral_storage_path) / batches_dir

            self.metadata = self._load_data(file = 'metadata', fold = fold, ephemeral_storage_path = ephemeral_storage_path, bucket = bucket, batches_dir = batches_dir, file_type = 'json')
            self.scalers = self._load_data(file = 'scalers', fold = fold, ephemeral_storage_path = ephemeral_storage_path, bucket = bucket, batches_dir = batches_dir, file_type = 'pkl')
            all_batches = self._load_data(file = 'batches', fold = fold, ephemeral_storage_path = ephemeral_storage_path, bucket = bucket, batches_dir = batches_dir, file_type = 'pkl')

        else:
            logger.info(f"Loading data from local directory {batches_dir}")
            # Load metadata
            metadata_path = self.batches_dir / f'fold_{fold}_metadata.json'
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Load scalers
            scalers_path = self.batches_dir / f'fold_{fold}_scalers.pkl'
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
            
            # Load all batches
            batches_path = self.batches_dir / f'fold_{fold}_batches.pkl'
            with open(batches_path, 'rb') as f:
                all_batches = pickle.load(f)
        
        # Filter subjects if specified
        if subjects is not None:
            available_subjects = list(all_batches.keys())
            subjects = [s for s in subjects if s in available_subjects]
            if not subjects:
                raise ValueError(f"No valid subjects found. Available subjects: {available_subjects}")
            all_batches = {subject: all_batches[subject] for subject in subjects}

        memory_info = estimate_device_memory_usage(batches_dir, fold, subjects, all_batches)
        logger.info(f"Estimated device memory usage: {memory_info['total_memory_gb']:.2f} GB")
        logger.info(f"Memory per batch: {memory_info['memory_per_batch_mb']:.2f} MB")
        
        # Flatten all batches into a single list and convert to tensors
        self.batches = []
        self.subject_batch_indices = {}
        
        start_idx = 0
        logger.info(f"Converting {len(all_batches)} subjects to tensors...")
        
        for subject_id, subject_batches in all_batches.items():
            logger.info(f"Processing subject {subject_id} with {len(subject_batches)} batches")
            
            for batch_data in subject_batches:
                if batch_data['X'].shape[0] != self.batch_size:
                    #### only use full batches
                    continue

                # Convert to tensors
                X = torch.from_numpy(batch_data['X']).float()
                y = torch.from_numpy(batch_data['y']).float()
                ##### preconvert weight to [B, 460, 1]
                weight = torch.from_numpy(np.full((X.shape[0], X.shape[1], 1), batch_data['weight'] / 100)).float()
                
                # Transfer to device if specified
                if device is not None:
                    X = X.to(device)
                    y = y.to(device)
                    weight = weight.to(device)

                # Store batch data
                processed_batch = {
                    'X': X,
                    'y': y,
                    'y_timestamps': batch_data['y_timestamps'],
                    'x_timestamps': batch_data['x_timestamps'],
                    'foot_id': batch_data['foot_id'],
                    'weight': weight,
                    'subject_id': batch_data['subject_id']
                }
                
                self.batches.append(processed_batch)
            
            end_idx = start_idx + len(subject_batches)
            self.subject_batch_indices[subject_id] = (start_idx, end_idx)
            start_idx = end_idx
        
        logger.info(f"Loaded {len(self.batches)} batches for {len(all_batches)} subjects")
        if device is not None:
            logger.info(f"All data transferred to device: {device}")
        
        # Store configuration
        self.config = self.metadata['config']

    def _load_data(self, file: str, fold: int, ephemeral_storage_path: Path, bucket: str, batches_dir: str, file_type: str):
        try:
            file_name = f'fold_{fold}_{file}.{file_type}'
            ephemeral_storage_path_file = ephemeral_storage_path / file_name
            if ephemeral_storage_path_file.exists():
                logger.info(f'Loading {file} from ephemeral storage')
                if file_type == 'json':
                    with open(ephemeral_storage_path_file, 'r') as f:
                        out = json.load(f) 
                elif file_type == 'pkl':
                    with open(ephemeral_storage_path_file, 'rb') as f:
                        out = pickle.load(f)
            else:
                logger.info(f'{file} does not exist in ephemeral storage, loading from S3')
                loader = S3DataLoader(bucket)
                ephemeral_storage_path.mkdir(exist_ok=True, parents=True)
                if file_type == 'json':
                    out = loader.load_json(os.path.join(f'force_modelling/{batches_dir}/', file_name))
                    logger.info(f'Saving {file} to ephemeral storage for future runs')
                    with open(ephemeral_storage_path_file, 'w') as f:
                        json.dump(out, f)
                elif file_type == 'pkl':
                    out = loader.load_pickle(os.path.join(f'force_modelling/{batches_dir}/', file_name))
                    logger.info(f'Saving {file} to ephemeral storage for future runs')
                    with open(ephemeral_storage_path_file, 'wb') as f:
                        pickle.dump(out, f)
            return out
        except Exception as e:
            logger.error(f'Failed to load {file} for fold {fold}')
            breakpoint()
            raise
            return None

    def __len__(self) -> int:
        """Return the number of batches"""
        return len(self.batches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray, float, str]:
        """
        Get a batch by index
        
        Returns:
            Tuple of (X, y, y_timestamps, x_timestamps, foot_id, weight, subject_id)
        """
        batch_data = self.batches[idx]
        
        return (
            batch_data['X'],
            batch_data['y'],
            batch_data['y_timestamps'],
            batch_data['x_timestamps'],
            batch_data['foot_id'],
            batch_data['weight'],
            batch_data['subject_id']
        )
    
    def get_subject_batches(self, subject_id: str) -> List[int]:
        """Get all batch indices for a specific subject"""
        if subject_id not in self.subject_batch_indices:
            return []
        
        start_idx, end_idx = self.subject_batch_indices[subject_id]
        return list(range(start_idx, end_idx))
    
    def get_subjects(self) -> List[str]:
        """Get list of all subjects in the dataset"""
        return list(self.subject_batch_indices.keys())
    
    def get_scalers(self) -> Dict:
        """Get the data scalers"""
        return self.scalers


def create_device_ram_data_loaders(
    fold: int,
    batches_dir: str,
    train_subjects: List[str],
    val_subjects: List[str],
    batch_size: int,
    device: Optional[torch.device] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    bucket: str = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for device RAM-based datasets
    
    Args:
        batches_dir: Directory containing the prepared batches
        train_subjects: List of training subject IDs
        val_subjects: List of validation subject IDs
        batch_size: Batch size (note: this is ignored since batches are pre-prepared)
        device: Device to transfer data to. If None, data stays on CPU.
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes (0 for single process)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = DeviceRAMForceDataset(batches_dir, fold, batch_size, train_subjects, device, bucket)
    val_dataset = DeviceRAMForceDataset(batches_dir, fold, batch_size, val_subjects, device, bucket)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,  # Each item is already a batch
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False if device is not None else True,  # No need for pin_memory if data is already on device
        collate_fn=lambda x: x[0]  # Unpack the batch since each item is already a batch
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # Each item is already a batch
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False if device is not None else True,  # No need for pin_memory if data is already on device
        collate_fn=lambda x: x[0]  # Unpack the batch since each item is already a batch
    )
    
    return train_loader, val_loader


def estimate_device_memory_usage(batches_dir: str, fold: int, subjects: Optional[List[str]] = None, all_batches: Dict[str, List[Dict[str, Any]]] = None) -> Dict[str, float]:
    """
    Estimate memory usage for loading batches to device
    
    Args:
        batches_dir: Directory containing the prepared batches
        subjects: List of subject IDs to estimate for. If None, estimates for all subjects.
    
    Returns:
        Dictionary with memory estimates in GB
    """
    # if bucket:
    #     loader = S3DataLoader(bucket_name=bucket)
    #     all_batches = loader.load_pickle(f'force_modelling/{batches_dir}/fold_{fold}_batches.pkl')
    # else:
    #     batches_path = Path(batches_dir) / f'fold_{fold}_batches.pkl'
        
    #     with open(batches_path, 'rb') as f:
    #         all_batches = pickle.load(f)
    
    # # Filter subjects if specified
    # if subjects is not None:
    #     available_subjects = list(all_batches.keys())
    #     subjects = [s for s in subjects if s in available_subjects]
    #     all_batches = {subject: all_batches[subject] for subject in subjects}
    
    total_memory_bytes = 0
    total_batches = 0
    
    for subject_id, subject_batches in all_batches.items():
        for batch_data in subject_batches:
            # Estimate memory for X and y tensors (float32)
            X_size = batch_data['X'].nbytes
            y_size = batch_data['y'].nbytes
            total_memory_bytes += X_size + y_size
            total_batches += 1
    
    # Convert to GB
    memory_gb = total_memory_bytes / (1024**3)
    
    return {
        'total_memory_gb': memory_gb,
        'total_batches': total_batches,
        'memory_per_batch_mb': (total_memory_bytes / total_batches) / (1024**2) if total_batches > 0 else 0
    } 