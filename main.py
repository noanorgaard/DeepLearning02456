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
        self.num_of_rows_in_train = 1000
# Usage
hparams = Hyperparameters()

# Load the data
path_to_data = "tmp/Data/ebnerd_small"
path_to_embeddings = "tmp/Data/Ekstra_Bladet_contrastive_vector/Ekstra_Bladet_contrastive_vector"

data = dataloader.Data(f"{path_to_data}/train/behaviors.parquet", f"{path_to_data}/articles.parquet", f"{path_to_data}/train/history.parquet", f"{path_to_embeddings}/contrastive_vector.parquet", hparams)



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

data.df_train = data.df_train.to_pandas()
data.df_train = data.df_train[data.df_train["article_id_fixed"].apply(lambda x: 0 not in x)]
his_input_np = data.df_train["article_id_fixed"].head(hparams.num_of_rows_in_train).to_numpy()

his_input_title = torch.empty(hparams.num_of_rows_in_train, hparams.history_size, hparams.title_size)
for i in range(hparams.num_of_rows_in_train):
    his_input_title[i] = torch.tensor(np.array([data.article_embeddings_dict[int(article_id)] for article_id in his_input_np[i]]))


pred_input_np = data.df_train["article_ids_inview"].head(hparams.num_of_rows_in_train).to_numpy()

pred_input_title = torch.empty(hparams.num_of_rows_in_train, hparams.negative_sampling_ratio+1, hparams.title_size)
for i in range(hparams.num_of_rows_in_train):
    pred_input_title[i] = torch.tensor(np.array([data.article_embeddings_dict[int(article_id)] for article_id in pred_input_np[i]]))


labels_np = data.df_train["labels"].head(hparams.num_of_rows_in_train).to_numpy()

labels = torch.empty(hparams.num_of_rows_in_train, hparams.negative_sampling_ratio+1)

for i in range(hparams.num_of_rows_in_train):
    labels[i] = torch.tensor(np.array([float(label) for label in labels_np[i]]))


dataset = TensorDataset(his_input_title, pred_input_title, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Train the model
train(nrms_model, dataloader, hparams.loss_func, optimizer, num_epochs=5, hparams=hparams)





# Train the model
#train(nrms_model, dataloader, criterion, optimizer, num_epochs=10, hparams=hparams)


# Print model summary (optional)
#print(model)














