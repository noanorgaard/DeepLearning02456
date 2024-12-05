import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from nrms import NRMS
from sklearn.metrics import roc_auc_score

# Define hyperparameters


from torch.utils.data import DataLoader, TensorDataset


def train(model, dataloader, val_dataloader, loss_func, optimizer, num_epochs, hparams):
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
        auc = 0
        for i,label_true in enumerate(all_labels):
            auc += roc_auc_score(label_true, all_outputs[i])
        auc /= len(all_labels)
        print(f"Epoch [{epoch + 1}/{num_epochs}], AUC: {auc:.4f}")

        # Validation step
        model.eval()
        val_labels = []
        val_outputs = []
        with torch.no_grad():
            for his_input_title, pred_input_title, labels in val_dataloader:
                outputs = model(his_input_title, pred_input_title)
                val_labels.extend(labels.cpu().numpy())
                val_outputs.extend(outputs.detach().cpu().numpy())

        # Calculate validation AUC
        val_auc = 0
        for i, label_true in enumerate(val_labels):
            val_auc += roc_auc_score(label_true, val_outputs[i])
        val_auc /= len(val_labels)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation AUC: {val_auc:.4f}")

        model.train()
