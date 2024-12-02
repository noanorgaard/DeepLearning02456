import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path

from ebrec.utils._behaviors import (create_binary_labels_column, truncate_history, sampling_strategy_wu2019)

# TODO join behaviors with history

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
                 path_to_data: str,
                 path_to_embedding: str,
                 hparams: None,
                 unknown_representation: str = "zeros",
                 eval_mode: bool = False):

        # Main data frames and embeddings
        self.behaviors = pl.scan_parquet(
            Path.joinpath(path_to_data, "train", "behaviors.parquet")
        )
        self.history = pl.scan_parquet(
            Path.joinpath(path_to_data, "train", "history.parquet")
        )
        self.articles = pd.read_parquet(
            Path.joinpath(path_to_data, "articles.parquet")
        )
        self.article_embeddings = pd.read_parquet(
            Path.joinpath(path_to_embedding, "contrastive_vector.parquet")
        )

        # Create a dictionary for article embeddings
        self.article_embeddings_dict = {row["article_id"]: row["contrastive_vector"] for row in self.article_embeddings.to_dict(orient='records')}

        # Hyperparameters
        self.hparams = hparams

        # make dataframe for training
        COLUMNS_FROM_HISTORY = ["user_id", "article_id_fixed"]
        COLUMNS_FROM_BEHAVIORS = ["user_id", "impression_id", "impression_time", "article_ids_clicked", "article_ids_inview"]

        # Columns we use
        self.history_col = "his_article_ids"
        self.inview_col = "article_ids_inview"
        self.labels_col = "labels"
        self.user_col = "user_id"

        # To be used for data representation
        self.unknown_representation = unknown_representation
        self.eval_mode = eval_mode
        self.unknown_index = [0]

        # Create lookup objects
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_embeddings_dict, unknown_representation=self.unknown_representation
        )

        self._prepare_history(COLUMNS_FROM_HISTORY)
        self._prepare_behaviors(COLUMNS_FROM_BEHAVIORS)
        self._join_dataframes()

        # Load data
        self.X, self.y = self._split_to_Xy()

    def _prepare_history(self, keep_columns):
        self.history = (
            self.history
            .select(keep_columns)                               # Selecting the columns we want to keep
            .rename({"article_id_fixed": "his_article_ids"})    # using prefix: his_ to indicate origin: history
            .collect()
            .pipe(                                                         # Truncating the history
                    truncate_history,
                    column="his_article_ids",
                    history_size=self.hparams.history_size,
                    padding_value=0,
                    enable_warning=False,
                )
            )

    def _prepare_behaviors(self, keep_columns):
        self.behaviors = (
            self.behaviors
            .select(keep_columns) # selecting the columns we want to keep
            .collect()
            .with_columns(
                length=pl.col('article_ids_clicked').map_elements(lambda x: len(x)))  # adding a column with the length of the clicked articles
            .filter(pl.col('length') == 1)  # we only want users with exactly one click in their impression
            .pipe(sampling_strategy_wu2019, npratio=self.hparams.negative_sampling_ratio, shuffle=True, clicked_col="article_ids_clicked",
                inview_col="article_ids_inview", with_replacement=False, seed=self.hparams.seed)   # down-sampling
            .pipe(create_binary_labels_column, clicked_col="article_ids_clicked",      # creating the binary labels column
                inview_col="article_ids_inview")
            .drop("length")
            )

    def _join_dataframes(self):
        ''''''
        self.df_train = (
            self.behaviors
            .join(self.history, on="user_id", how="left") # joining the history and behaviors dataframes
            .head(self.hparams.num_of_rows_in_train)
        )

    def _split_to_Xy(self):
        ''' Split data frame into X and y '''
        X = self.df_train.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.df_train[self.labels_col]
        return X, y

    def map_to_embeddings(self, df: pl.DataFrame) -> pl.DataFrame:
        ''' Map the article ids in history_col and inview_col to their embeddings '''
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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        Xi = self.X[idx].pipe(self.map_to_embeddings)
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


# Create a PyTorch DataLoader
def create_dataloader(
    path_to_data: str,
    path_to_embedding: str,
    hparams,
    batch_size: int = 32,
    unknown_representation: str = "zeros",
    eval_mode: bool = False):

    dataset = NewsrecDataset(path_to_data, path_to_embedding, hparams, unknown_representation, eval_mode)

    return DataLoader(dataset, batch_size=batch_size, shuffle=not eval_mode)




