import pandas as pd
import polars as pl
import pyarrow
from pathlib import Path
import numpy as np
from ebrec.utils._behaviors import truncate_history
import torch


def create_test_set(path_to_data, path_to_embeddings, hparams):
    """
    Create a test set from the given data and embeddings
    """
    # Load the data
    behaviors = pd.read_parquet(path_to_data / "test" / "behaviors.parquet")
    history = pl.read_parquet(path_to_data / "test" / "history.parquet")
    article_embeddings = pd.read_parquet(path_to_embeddings / "contrastive_vector.parquet")

    # Create a dictionary for article embeddings
    article_embeddings_dict = {row["article_id"]: row["contrastive_vector"] for row in
                               article_embeddings.to_dict(orient='records')}
    # Add the zero vector for article id 0 (padding)
    article_embeddings_dict[0] = np.zeros(hparams.title_size)

    # Only keep impression_id, user_id, article_ids_inview
    behaviors = behaviors[["impression_id", "user_id", "article_ids_inview"]]

    history = prepare_history(history, ["user_id", "article_id_fixed"], hparams)
    history = history.to_pandas()

    # Merge the two dataframe on user_id
    data = behaviors.merge(history, on="user_id", how="inner", sort = False)
    data.set_index("impression_id", inplace=True)

    return data, article_embeddings_dict


def prepare_history(history, keep_columns, hparams):
    history_new = (
        history
        .select(keep_columns)  # Selecting the columns we want to keep
        .rename({"article_id_fixed": "his_article_ids"})  # using prefix: his_ to indicate origin: history
        .pipe(  # Truncating the history
            truncate_history,
            column="his_article_ids",
            history_size=hparams.history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    return history_new

