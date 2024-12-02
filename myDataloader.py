from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import polars as pl

from ebrec.utils._articles_behaviors import map_list_article_id_to_value
from ebrec.utils._python import (
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)

class NewsrecDataset(Dataset):
    def __init__(self,
                 df: pl.DataFrame,
                 history_col: str,
                 article_dict: dict[int, any],
                 history_size: int = 30,
                 embedding_size: int = 768,
                 unknown_representation: str = "zeros",
                 eval_mode: bool = False,
                 inview_col: str = "article_ids_inview",
                 labels_col: str = "labels",
                 user_col: str   = "user_id"):

        self.df = df
        self.history_col = history_col
        self.article_dict = article_dict
        self.history_size = history_size
        self.embedding_size = embedding_size
        self.unknown_representation = unknown_representation
        self.eval_mode = eval_mode
        self.inview_col = inview_col
        self.labels_col = labels_col
        self.user_col = user_col

        # Create lookup objects
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]

        # Load data
        self.X, self.y = self.load_data()

    def load_data(self):
        X = self.df.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.df[self.labels_col]
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        Xi = self.X[idx].pipe(self.transform)
        yi = self.y[idx]

        if self.eval_mode:
            repeats = np.array(Xi["n_samples"])
            yi = np.array(yi.explode().to_list()).reshape(-1,1)
            his_input_embeddings = repeat_by_list_values_from_matrix(
                Xi[self.history_col].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            pred_input_embeddings = self.lookup_article_matrix[
                Xi[self.inview_col].explode().to_list()
            ]
        else:
            yi = np.array(yi.to_list())
            his_input_embeddings = self.lookup_article_matrix[
                Xi[self.history_col].to_list()
            ]
            pred_input_embeddings = self.lookup_article_matrix[
                Xi[self.inview_col].to_list()
            ]
            #pred_input_embeddings = pred_input_embeddings.reshape(-1,1,self.embedding_size)

        # Make into torch tensors
        his_input_embeddings = torch.tensor(his_input_embeddings, dtype=torch.float32)
        pred_input_embeddings = torch.tensor(pred_input_embeddings, dtype=torch.float32)
        yi = torch.tensor(yi,dtype=torch.float32)

        # Reshape
        his_input_embeddings = his_input_embeddings.view(-1,self.embedding_size)
        pred_input_embeddings = pred_input_embeddings.view(-1,self.embedding_size)

        return his_input_embeddings, pred_input_embeddings, yi

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )

# Create a PyTorch DataLoader
def create_dataloader(df, history_column, article_dict, history_size, embedding_dim, unknown_representation,
                      eval_mode=False, batch_size=32, **kwargs):
    dataset = NewsrecDataset(df, history_column, article_dict, history_size, embedding_dim, unknown_representation, eval_mode, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=not eval_mode)
