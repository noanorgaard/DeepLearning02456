import numpy as np
import polars as pl



class Data:
    def __init__(self, path_to_behaviors, path_to_articles, path_to_history, path_to_embeddings):
        self.behaviors = pl.read_parquet(path_to_behaviors)
        self.articles = pl.read_parquet(path_to_articles)
        self.history = pl.read_parquet(path_to_history)
        self.article_embeddings = pl.read_parquet(path_to_embeddings)
        self.article_dict = {row["article_id"]: row["contrastive_vector"] for row in self.article_embeddings.to_dicts()}







