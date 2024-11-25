import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from nrms import NRMS
from sklearn.metrics import roc_auc_score

# Define hyperparameters


from torch.utils.data import DataLoader, TensorDataset


def train(model, dataloader, loss_func, optimizer, num_epochs, hparams):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        all_labels = []
        all_outputs = []
        for i, (his_input_title, pred_input_title, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(his_input_title, pred_input_title)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())

        # Calculate AUC
        auc = roc_auc_score(all_labels, all_outputs)
        print(f"Epoch [{epoch + 1}/{num_epochs}], AUC: {auc:.4f}")