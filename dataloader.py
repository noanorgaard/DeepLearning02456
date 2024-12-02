import numpy as np
import polars as pl
import pandas as pd
import pyarrow
import fastparquet

from ebrec.utils._behaviors import (create_binary_labels_column, truncate_history, sampling_strategy_wu2019)

# TODO join behaviors with history

class Data:
    def __init__(self, path_to_behaviors, path_to_articles, path_to_history, path_to_embeddings, params):
        self.behaviors = pl.read_parquet(path_to_behaviors)
        self.articles = pd.read_parquet(path_to_articles)
        self.history = pl.read_parquet(path_to_history)
        self.article_embeddings = pd.read_parquet(path_to_embeddings)
        self.article_embeddings_dict = {row["article_id"]: row["contrastive_vector"] for row in self.article_embeddings.to_dict(orient='records')}
        self.hparams = params

        # make dataframe for training
        COLUMNS_FROM_HISTORY = ["user_id", "article_id_fixed"]
        COLUMNS_FROM_BEHAVIORS = ["user_id", "impression_id", "impression_time", "article_ids_clicked", "article_ids_inview"]

        self._prepare_history(COLUMNS_FROM_HISTORY)
        self._prepare_behaviors(COLUMNS_FROM_BEHAVIORS)
        self._join_dataframes()

    def _prepare_history(self, keep_columns):
        self.history = (
            self.history
            .select(keep_columns)                               # Selecting the columns we want to keep
            .rename({"article_id_fixed": "his_article_ids"})    # using prefix: his_ to indicate origin: history
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
            .with_columns(
                length=pl.col('article_ids_clicked').map_elements(lambda x: len(x)))  # adding a column with the length of the clicked articles
            .filter(pl.col('length') == 1)  # we only want users with exactly one click in their impression
            .pipe(sampling_strategy_wu2019, npratio=self.haparams.negative_sampling_ratio, shuffle=True, clicked_col="article_ids_clicked",
                inview_col="article_ids_inview", with_replacement=False, seed=self.hparams.seed)   # down-sampling
            .pipe(create_binary_labels_column, clicked_col="article_ids_clicked",      # creating the binary labels column
                inview_col="article_ids_inview")
            .drop("length")
            )

    def _join_dataframes(self):
        self.df_train = (
            self.behaviors
            .join(self.history, on="user_id", how="left") # joining the history and behaviors dataframes
        )




