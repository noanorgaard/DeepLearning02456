import dataloader
from nrms import NRMS
from train import train, HParams
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from  train import generate_dummy_data


# Load the data
path_to_data = "tmp/Data/ebnerd_small"
path_to_embeddings = "tmp/Data/Ekstra_Bladet_contrastive_vector/Ekstra_Bladet_contrastive_vector"

data = dataloader.Data(f"{path_to_data}/train/behaviors.parquet", f"{path_to_data}/articles.parquet", f"{path_to_data}/train/history.parquet", f"{path_to_embeddings}/contrastive_vector.parquet")


# Define hyperparameters
class Hyperparameters:
    def __init__(self, data):
        self.history_size = 30
        self.title_size = 100
        self.head_num = 8
        self.head_dim = 16
        self.attention_hidden_dim = 200
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.negative_samopling_ratio = 4

# Usage
hparams = Hyperparameters(data)

# Initialize NRMS model
word2vec_embedding = data.article_embeddings["contrastive_vector"].to_numpy()

# Build NRMS model
model = NRMS(hparams, word2vec_embedding)

his_input_title = data.history["article_id_fixed"].to_numpy()
clicked_title_in_impression = data.behaviors["article_ids_clicked"].to_numpy()
in_view_title_in_impression = data.behaviors["article_ids_inview"].to_numpy()
labels = data.behaviors["label"].to_numpy()




# Train the model
train(nrms_model, dataloader, criterion, optimizer, num_epochs=10, hparams=hparams)


# Print model summary (optional)
print(model)













