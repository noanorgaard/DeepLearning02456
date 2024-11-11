import dataloader

# Load the data
path_to_data = "tmp/Data/ebnerd_small"
path_to_embeddings = "tmp/Data/Ekstra_Bladet_contrastive_vector/Ekstra_Bladet_contrastive_vector"

data = dataloader.Data(f"{path_to_data}/train/behaviors.parquet", f"{path_to_data}/articles.parquet", f"{path_to_data}/train/history.parquet", f"{path_to_embeddings}/contrastive_vector.parquet")



