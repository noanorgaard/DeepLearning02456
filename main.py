import dataloader
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
from myDataloader import create_dataloader



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

# Load the data
path_to_data = "tmp/Data/ebnerd_small"
path_to_embeddings = "tmp/Data/Ekstra_Bladet_contrastive_vector/Ekstra_Bladet_contrastive_vector"

data = dataloader.Data(f"{path_to_data}/train/behaviors.parquet", f"{path_to_data}/articles.parquet", f"{path_to_data}/train/history.parquet", f"{path_to_embeddings}/contrastive_vector.parquet", hparams)


# Initialize NRMS model
news_encoder = NewsEncoder(hparams, units_per_layer=[512, 512, 512])
nrms_model = NRMS(hparams, news_encoder)

# Define  optimizer
optimizer = optim.Adam(nrms_model.parameters(), lr=hparams.learning_rate)


dataloader_train = create_dataloader(data.df_train.head(hparams.num_of_rows_in_train), "his_article_ids", data.article_embeddings_dict, hparams.history_size, hparams.title_size, "zeros", eval_mode=False, batch_size=32)

# Train the model
train(nrms_model, dataloader_train, hparams.loss_func, optimizer, num_epochs=1, hparams=hparams)


torch.save(nrms_model, "models/nrms_model_test.pth")

# Load the model
nrms_model = torch.load("models/nrms_model_test.pth")
nrms_model.eval()
















