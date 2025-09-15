
import json
import os
import yaml
import time
import pickle
import gc
import random

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.amp as amp  # For mixed precision training
# import optuna

import numpy as np
import pandas as pd


# Configure logging
import logging
logger = logging.getLogger(__name__)


from src.device_ram_dataset import create_device_ram_data_loaders


def get_data_loaders(fold: int, config: Dict, device: torch.device, bucket: str) -> Tuple[DataLoader, DataLoader]:

    return create_device_ram_data_loaders(
                fold=fold,
                batches_dir=config['data']['batches_dir'],
                train_subjects=config['data']['train_subjects'],
                val_subjects=config['data']['val_subjects'],
                batch_size=config['training']['batch_size'],
                device=device,
                shuffle=True,
                num_workers=0,
                bucket=bucket
            )