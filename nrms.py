import torch
import torch.nn as nn
import torch.nn.functional as F



class NewsEncoder(nn.Module):
    def __init__(self, hparams, word2vec_embedding, seed=None):
        super(NewsEncoder, self).__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.tensor(word2vec_embedding, dtype=torch.float32), freeze=False)
        self.dropout = nn.Dropout(hparams.dropout)
        self.self_attention = nn.MultiheadAttention(embed_dim=hparams.head_dim, num_heads=hparams.head_num)
        self.att_layer = nn.Linear(hparams.head_dim, hparams.attention_hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, sequences_input_title):
        embedded_sequences_title = self.embedding_layer(sequences_input_title)
        y = self.dropout(embedded_sequences_title)
        y, _ = self.self_attention(y, y, y)
        y = self.dropout(y)
        pred_title = self.tanh(self.att_layer(y))
        return pred_title


class UserEncoder(nn.Module):
    def __init__(self, hparams, newsencoder, seed=None):
        super(UserEncoder, self).__init__()
        self.newsencoder = newsencoder
        self.dropout = nn.Dropout(hparams.dropout)
        self.self_attention = nn.MultiheadAttention(embed_dim=hparams.head_dim, num_heads=hparams.head_num)
        self.att_layer = nn.Linear(hparams.head_dim, hparams.attention_hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, his_input_title):
        batch_size, history_size, title_size = his_input_title.size()
        his_input_title = his_input_title.view(batch_size * history_size, title_size)
        click_title_presents = self.newsencoder(his_input_title)
        click_title_presents = click_title_presents.view(batch_size, history_size, -1)
        y, _ = self.self_attention(click_title_presents, click_title_presents, click_title_presents)
        y = self.dropout(y)
        user_present = self.tanh(self.att_layer(y))
        return user_present


class NRMS(nn.Module):
    def __init__(self, hparams, word2vec_embedding=None, seed=None):
        super(NRMS, self).__init__()
        self.hparams = hparams
        self.seed = seed
        self.word2vec_embedding = word2vec_embedding
        self.embedding_layer = nn.Embedding.from_pretrained(torch.tensor(word2vec_embedding, dtype=torch.float32), freeze=False)
        self.dropout = nn.Dropout(hparams.dropout)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=hparams.learning_rate)
        self.news_encoder = NewsEncoder(self.hparams, self.word2vec_embedding, self.seed)
        self.user_encoder = UserEncoder(self.hparams, self.news_encoder, self.seed)

    def forward(self, his_input_title, pred_input_title):
        user_present = self.user_encoder(his_input_title)
        news_present = self.news_encoder(pred_input_title)
        scores = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
        return scores