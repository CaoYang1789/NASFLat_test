# NASFlat Latency test
Based on the introduction of Flat https://github.com/abdelfattah-lab/nasflat_latency. It is necessary to download `nasflat_embeddings_04_03_24.zip` and `NASFLATBench_v1.pkl` and add them to the `/content` directory (Step 2 and Step 3 demonstrate how to upload file `nasflat_embeddings_04_03_24.zip` from Google Drive to Colab by copying it, The file `NASFLATBench_v1.pkl` is relatively small, so you can decide on your own how to move it to contentnasflat_latency/correlation_trainer.).
## Step 1: Clone the GitHub Repository

Clone the GitHub repository.

```bash
!git clone https://github.com/abdelfattah-lab/nasflat_latency.git
%cd nasflat_latency
```

## Step 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 3: Add Files to the Content Directory

```bash
%cd nasflat_latency
# Create the target folder
!mkdir -p nas_embedding_suite

# Copy the file to the target folder
!cp /content/drive/MyDrive/nasflat_embeddings_04_03_24.zip nas_embedding_suite/

# Unzip the file
!unzip nas_embedding_suite/nasflat_embeddings_04_03_24.zip -d nas_embedding_suite/

# Create the target folder (if not already created)
!mkdir -p nas_embedding_suite/NDS/nds_data

# Move NASFLATBench_v1.pkl to the specified directory
!mv /content/NASFLATBench_v1.pkl nas_embedding_suite/NDS/nds_data/
```

## Step 4: Modify `main_trf.py` Header

Edit the header of `contentnasflat_latency/correlation_trainer/main_trf.py` as follows(The complete code has been uploaded):

```python
import os
import sys  # We added
sys.path.append("/content/nasflat_latency")  # We added

BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
from scipy.stats import spearmanr, kendalltau
import torch
from torch.utils.data import DataLoader
import argparse, sys, time, random, os
import numpy as np
from pprint import pprint
from tqdm import tqdm
from utils import CustomDataset, get_tagates_sample_indices
from torch.optim.lr_scheduler import StepLR
from pprint import pprint
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
# sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')  # We removed
```

## Step 5: Set Environment Variables

Make sure to manually set the environment variable.

```python
import os

# Set environment variable
os.environ['PROJ_BPATH'] = "/content/nasflat_latency"
print(os.environ.get('PROJ_BPATH'))  # Should output "/content/nasflat_latency"
```

## Step 6: Test Import

Load the `pkl` file and test the import.

```python
import sys
sys.path.append("/content/nasflat_latency")

# Test the import
try:
    from nas_embedding_suite.fbnet_ss import FBNet
    print("Import successful!")
except ModuleNotFoundError as e:
    print("Import failed:", e)
```

## Step 7: Clone NAS-Bench-201 Repository

Clone the `NAS-Bench-201` repository and supplement the missing content in the project.

```bash
!git clone https://github.com/D-X-Y/NAS-Bench-201.git /content/temp_nas_bench_201

# View the folder contents
!ls /content/temp_nas_bench_201

# Copy missing content
!cp -r /content/temp_nas_bench_201/nas_201_api /content/nasflat_latency/nas_embedding_suite/

# View contents again
!ls /content/nasflat_latency/nas_embedding_suite
```

## Step 8: Modify `nb201_ss.py`

Modify the header of `nasflat_latency/nas_embedding_suite/nb201_ss.py` as follows(The complete code has been uploaded.):

```python
import torch
import json
from tqdm import tqdm
import types
import copy
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing
import pickle
import random, time

import sys, os

sys.path.append(os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite')
from nb123.nas_bench_201.cell_201 import Cell201

sys.path.append("/content/nasflat_latency/nas_embedding_suite")

from nas_201_api import NASBench201API as NB2API

BASE_PATH = os.environ['PROJ_BPATH'] + "/" + 'nas_embedding_suite/embedding_datasets/'
```

## Step 9: Train FBNet with CAZ Sampler

Run the following command to train FBNet using the CAZ sampler.

```bash
!python /content/nasflat_latency/correlation_trainer/main_trf.py --seed 42 --name_desc study_6_5_f_zcp --sample_sizes 800 --task_index 5 --representation adj_gin_zcp --num_trials 1 --transfer_sample_sizes 20 --transfer_lr 0.001 --transfer_epochs 30 --transfer_hwemb --space fbnet --gnn_type ensemble --sampling_metric a2vcatezcp --ensemble_fuse_method add
```
## Step 10: Result example
![image](https://github.com/user-attachments/assets/ebd4ff2f-bb30-40cc-a31b-0cb0b3274d15)


