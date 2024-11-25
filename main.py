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


# Load the data
path_to_data = "tmp/Data/ebnerd_small"
path_to_embeddings = "tmp/Data/Ekstra_Bladet_contrastive_vector/Ekstra_Bladet_contrastive_vector"

data = dataloader.Data(f"{path_to_data}/train/behaviors.parquet", f"{path_to_data}/articles.parquet", f"{path_to_data}/train/history.parquet", f"{path_to_embeddings}/contrastive_vector.parquet")


# Define hyperparameters
class Hyperparameters:
    def __init__(self, data):
        self.history_size = 20
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
# Usage
hparams = Hyperparameters(data)

# Initialize NRMS model
#word2vec_embedding = data.article_embeddings["contrastive_vector"].to_numpy()

# Make news encoder and test it
news_encoder = NewsEncoder(hparams, units_per_layer=[512, 512, 512])
print(news_encoder)
article_ids = data.articles["article_id"].head(5)
articles = []
for article_id in article_ids:
    articles.append(data.article_embeddings_dict[article_id])

articles = np.array(articles)
articles_tensor = torch.from_numpy(articles)
# Make a forward pass
encoded_article = news_encoder(articles_tensor)
print(encoded_article)


# Initialize NRMS model
news_encoder = NewsEncoder(hparams, units_per_layer=[512, 512, 512])
nrms_model = NRMS(hparams, news_encoder)

# Prepare dummy input data
batch_size = 2
his_input_title = torch.randn(batch_size, hparams.history_size, hparams.title_size)
pred_input_title = torch.randn(batch_size, hparams.negative_sampling_ratio + 1, hparams.title_size)
labels = torch.zeros(batch_size, hparams.negative_sampling_ratio + 1)
for i in range(batch_size):
    labels[i, torch.randint(0, hparams.negative_sampling_ratio + 1, (1,))] = 1
labels = labels.float()

# Make a forward pass
preds = nrms_model(his_input_title, pred_input_title)
print(preds)



# Create DataLoader
dataset = TensorDataset(his_input_title, pred_input_title, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define  optimizer
optimizer = optim.Adam(nrms_model.parameters(), lr=hparams.learning_rate)


# Train the model
train(nrms_model, dataloader, hparams.loss_func, optimizer, num_epochs=10, hparams=hparams)





# Train the model
#train(nrms_model, dataloader, criterion, optimizer, num_epochs=10, hparams=hparams)


# Print model summary (optional)
#print(model)














