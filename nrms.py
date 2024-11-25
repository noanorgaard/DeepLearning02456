import torch
import torch.nn as nn
import torch.nn.functional as F

# Taken from https://stackoverflow.com/questions/62912239/tensorflows-timedistributed-equivalent-in-pytorch
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y



class AttentionLayer2(nn.Module):
    def __init__(self, hparams, seed=0):
        super(AttentionLayer2, self).__init__()
        dim = hparams.dim_attention_later2
        self.seed = seed
        self.W = nn.Parameter(torch.Tensor(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.q = nn.Parameter(torch.Tensor(dim, 1))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.q, gain=nn.init.calculate_gain('tanh'))

    def forward(self, inputs, mask=None):
        attention = torch.tanh(torch.matmul(inputs, self.W) + self.b)
        attention = torch.matmul(attention, self.q).squeeze(-1)

        if mask is None:
            attention = torch.exp(attention)
        else:
            attention = torch.exp(attention) * mask.float()

        attention_weight = attention / (torch.sum(attention, dim=-1, keepdim=True) + 1e-8)
        attention_weight = attention_weight.unsqueeze(-1)
        weighted_input = inputs * attention_weight
        return torch.sum(weighted_input, dim=1)


class NewsEncoder(nn.Module):
    def __init__(self, hparams, units_per_layer=[512, 512, 512]):
        super(NewsEncoder, self).__init__()
        self.hparams = hparams
        self.document_vector_dim = hparams.title_size
        self.output_dim = hparams.head_num * hparams.head_dim

        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.document_vector_dim, num_heads=hparams.head_num)


        layers = []
        input_dim = self.document_vector_dim
        for units in units_per_layer:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.Dropout(hparams.dropout))
            input_dim = units

        layers.append(nn.Linear(input_dim, self.output_dim))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        attn_output, _ = self.multihead_attention(x, x, x)
        return self.model(x)


class UserEncoder(nn.Module):
    def __init__(self, hparams, newsencoder):
        super(UserEncoder, self).__init__()
        self.newsencoder = TimeDistributed(newsencoder, batch_first=True)
        self.newsencoder_output_dim = hparams.newsencoder_output_dim
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.newsencoder_output_dim, num_heads=hparams.head_num)
        self.attention_layer = AttentionLayer2(hparams)



    def forward(self, x):
        # Encode the news history
        encoded_news = self.newsencoder(x)

        # Apply multi-head attention
        attn_output, _ = self.multihead_attention(encoded_news, encoded_news, encoded_news)

        # Apply the attention layer
        user_representation = self.attention_layer(attn_output)

        return user_representation


class NRMS(nn.Module):
    def __init__(self, hparams, newsencoder):
        super(NRMS, self).__init__()
        self.hparams = hparams
        self.newsencoder = newsencoder
        self.userencoder = UserEncoder(hparams, newsencoder)

    def forward(self, his_input_title, pred_input_title):
        # Encode the user history
        user_present = self.userencoder(his_input_title)  # u vector

        # Encode the predicted titles
        batch_size, num_titles, title_size = pred_input_title.size()
        pred_input_title = pred_input_title.view(-1, title_size)
        news_present = self.newsencoder(pred_input_title)  # r vector
        news_present = news_present.view(batch_size, num_titles, -1)

        # Compute dot product and apply softmax
        preds = torch.matmul(news_present, user_present.unsqueeze(-1)).squeeze(-1)
        preds = F.softmax(preds, dim=-1)

        return preds

    def score(self, his_input_title, pred_input_title_one):
        # Encode the user history
        user_present = self.userencoder(his_input_title)  # u vector

        # Encode the single predicted title
        pred_title_one_reshape = pred_input_title_one.view(-1, self.hparams.title_size)
        news_present_one = self.newsencoder(pred_title_one_reshape)  # r vector for one article

        # Compute dot product and apply sigmoid
        pred_one = torch.matmul(news_present_one, user_present.unsqueeze(-1)).squeeze(-1)
        pred_one = torch.sigmoid(pred_one)

        return pred_one