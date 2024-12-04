from nrms import NRMS
import pandas as pd
import polars as pl
from train import train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from nrms import NewsEncoder
import torch.optim as optim
from dataloader import create_dataloader
from pathlib import Path


# Define hyperparameters
class Hyperparameters:
    def __init__(self):
        self.history_size = 30
        self.title_size = 768
        self.head_num = 16
        self.head_dim = 16
        self.attention_hidden_dim = 200
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.negative_sampling_ratio = 4
        self.newsencoder_output_dim = 256
        self.dim_attention_later2 = 256
        self.loss_func = nn.CrossEntropyLoss()
        self.seed = 123
        self.num_of_rows_in_train = 400
# Usage
hparams = Hyperparameters()

# Initialize NRMS model
news_encoder = NewsEncoder(hparams, units_per_layer=[512, 512, 512])
nrms_model = NRMS(hparams, news_encoder)

# Load the model
nrms_model = torch.load("models/nrms_model_5_epoch_100k.pth")
nrms_model.eval()

# Load the data
path_to_data = "tmp/ebnerd_testset/test"
path_to_embeddings = "tmp/Data/Ekstra_Bladet_contrastive_vector/Ekstra_Bladet_contrastive_vector"

test_data = create_test_set(Path(path_to_data), Path(path_to_embeddings),hparams)











