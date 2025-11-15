import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import functools
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns  # <-- This will now work
from typing import List, Tuple, Optional
import gc
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, GroupKFold  # <-- This will now work
from sklearn.metrics import roc_auc_score
import pydicom
import polars as pl

# The RSNA inference server (needed for submission)
try:
    import kaggle_evaluation.rsna_inference_server as rsna_inference_server
except ImportError:
    print("Inference server not found (OK for training)")

warnings.filterwarnings('ignore')

print(f"Imports successful. Numpy version: {np.__version__}")
