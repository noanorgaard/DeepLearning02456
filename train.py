import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from nrms import NRMS

# Define hyperparameters
class HParams:
    head_num = 8
    head_dim = 16
    attention_hidden_dim = 64
    dropout = 0.2
    learning_rate = 0.001
    history_size = 50
    title_size = 30

# Dummy data for demonstration purposes
def generate_dummy_data(num_samples, history_size, title_size):
    his_input_title = torch.randint(0, 1000, (num_samples, history_size, title_size))
    pred_input_title = torch.randint(0, 1000, (num_samples, title_size))
    labels = torch.randint(0, 2, (num_samples,))
    return his_input_title, pred_input_title, labels

# Training function
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for his_input_title, pred_input_title, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(his_input_title, pred_input_title)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



