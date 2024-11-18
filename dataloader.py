import numpy as np
import polars as pl
import pandas as pd
import pyarrow
import fastparquet



class Data:
    def __init__(self, path_to_behaviors, path_to_articles, path_to_history, path_to_embeddings):
        self.behaviors = pd.read_parquet(path_to_behaviors)
        self.articles = pd.read_parquet(path_to_articles)
        self.history = pd.read_parquet(path_to_history)
        self.article_embeddings = pd.read_parquet(path_to_embeddings)
        #self.article_embeddings_dict = {row["article_id"]: row["contrastive_vector"] for row in self.article_embeddings.to_dicts()}







