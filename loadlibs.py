import os
import sys
import glob
import cv2
sys.path.append("./apex")

# from apex import amp

from PIL import Image
import random
import pandas as pd
import numpy as np
import math
import datetime
from typing import Optional
from tqdm.auto import tqdm as tq
import warnings
warnings.filterwarnings(action='ignore') 
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, efficientnet_v2_l
from torchvision import transforms
# from transformers import logging
# from transformers import VisionEncoderDecoderModel, AutoTokenizer
# from transformers import TrOCRProcessor
# from transformers import MobileViTImageProcessor, MobileViTModel, MobileViTConfig
# logging.set_verbosity_error()
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.geometric.transforms import Affine as AF
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import yaml

