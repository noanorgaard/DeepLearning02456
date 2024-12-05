from nrms import NRMS
import torch
import torch.nn as nn
import numpy as np
from nrms import NewsEncoder
from pathlib import Path
from test_set_dataloader import create_test_set


# Define hyperparameters
class Hyperparameters:
    def __init__(self):
        self.history_size = 30
        self.title_size = 768
        self.head_num = 16
        self.head_dim = 16
        self.attention_hidden_dim = 200
        self.dropout = 0.2
        self.learning_rate = 0.0001
        self.negative_sampling_ratio = 4
        self.newsencoder_output_dim = 256
        self.dim_attention_later2 = 256
        self.loss_func = nn.CrossEntropyLoss()
        self.seed = 123
        self.weight_decay = 1e-4
        self.num_of_rows_in_train = 1000
# Usage
hparams = Hyperparameters()

# Initialize NRMS model
news_encoder = NewsEncoder(hparams, units_per_layer=[512, 512, 512])
nrms_model = NRMS(hparams, news_encoder)

# Load the model
nrms_model = torch.load("models/nrms_model_test.pth")
nrms_model.eval()

# Load the data
path_to_data = "tmp/ebnerd_testset"
path_to_embeddings = "tmp/Data/Ekstra_Bladet_contrastive_vector/Ekstra_Bladet_contrastive_vector"

test_data, article_embeddings_dict = create_test_set(Path(path_to_data), Path(path_to_embeddings),hparams)

folder_name = Path("test")
path_to_txt = Path("submission_folders" / folder_name)

# Create or overwrite a txt file
with open(path_to_txt / "predictions.txt", "w") as f:
    # Convert the article_ids_inview and his_article_ids to tensors with the embeddings
    for impression_id, row in test_data.iterrows():
        his_article_ids = row["his_article_ids"]
        article_ids_inview = row["article_ids_inview"]

        # Create row in tensor_data with index i and columns his_article_ids and article_ids_inview as tensors
        his_article_embeddings = np.array([article_embeddings_dict[article_id] for article_id in his_article_ids])
        his_article_tensor = torch.tensor(his_article_embeddings)

        article_ids_inview_embeddings = np.array(
            [article_embeddings_dict[article_id] for article_id in article_ids_inview])
        article_ids_inview_tensor = torch.tensor(article_ids_inview_embeddings)


        with torch.no_grad():
            predictions = nrms_model(his_article_tensor.unsqueeze(0), article_ids_inview_tensor.unsqueeze(0))
            predictions = predictions.squeeze(0)
            predictions = predictions.numpy()

        # Convert predictions into labels where the highest value is 1, second highest is 2 etc.
        labels = np.argsort(-predictions) + 1
        labels = labels.tolist()

        # Write impression_id and labels to file
        f.write(f"{impression_id} {' '.join(map(str, labels))}\n")

















