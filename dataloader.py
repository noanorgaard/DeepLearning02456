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

        title_in_impression = self.behaviors[["user_id", "article_ids_inview", "article_ids_clicked"]]
        title_in_impression_grouped_user_id = title_in_impression.set_index("user_id")

        # Remove duplicates in index by making new list of article_ids_inview and article_ids_clicked
        duplicates_bool = title_in_impression_grouped_user_id.index.duplicated(keep="first")
        new_df = title_in_impression_grouped_user_id[~duplicates_bool]
        df_duplicates = title_in_impression_grouped_user_id[duplicates_bool]
        for index, row in df_duplicates.iterrows():
            new_df.at[index, "article_ids_inview"] = list(
                set(np.append(new_df.loc[index, "article_ids_inview"], row["article_ids_inview"])))
            new_df.at[index, "article_ids_clicked"] = list(
                set(np.append(new_df.loc[index, "article_ids_clicked"], row["article_ids_clicked"])))

        new_df["article_ids_inview"].apply(tuple)
        new_df["article_ids_clicked"].apply(tuple)
        not_clicked = [
            set(row["article_ids_inview"]).difference(set(row["article_ids_clicked"]))
            for index, row in new_df.iterrows()
        ]
        new_df.loc[:, "article_ids_not_clicked"] = pd.Series(not_clicked, index=new_df.index)
        self.user_click_information = new_df







