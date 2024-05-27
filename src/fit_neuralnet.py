'''
This script fits a neural network over a grid of hyperparameters to the training data using 10-fold cross-validation with early stopping.
The type of features used as training data are either eigenvector centralities or time course averages, which can be selected from the terminal.
'''

from itertools import product
from pathlib import Path
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd



def input_parse():
    '''
    For parsing terminal arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", "-f", choices=['centrality', 'average'], required=True, help="Type of feature to use for training.")
    args = parser.parse_args()

    return args



class Model(nn.Module):
    def __init__(self, dropout=0.2, two_layers=False):
        super(Model, self).__init__()
        
        if two_layers:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(111, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(111, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        logit = self.linear_relu_stack(x)
        return logit



def train_model(model, train_loader, criterion, optimizer, device):
    '''
    Train model for one epoch. Returns the average loss across batches.
    '''
    model.train()
    epoch_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)

    return avg_loss



def evaluate_model(model, val_loader, criterion, device):
    '''
    Evaluate model on validation data. Returns the average loss and an F1 score.
    '''
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss = loss.item()
            wf1_score = f1_score(labels.cpu().numpy(), (outputs.cpu().numpy() > 0.5).astype(int), average='weighted')

    return val_loss, wf1_score



def early_stopping(train_loader, val_loader, model, criterion, optimizer, device, patience, delta):
    '''
    Implements early stopping during training. Returns the best model, based on validation loss, along with its training and validation loss. 
    '''
    best_loss = float('inf')
    best_model = None
    counter = 0
    loss_history = pd.DataFrame(columns=['train_loss', 'val_loss'])
    
    for epoch in tqdm(range(1, 101), desc='Training model'):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, wf1_score = evaluate_model(model, val_loader, criterion, device)

        # log training history
        loss_history.loc[len(loss_history.index)] = [train_loss, val_loss] 

        # check if validation loss improved
        if val_loss < best_loss - delta:
            best_loss = val_loss
            best_wf1 = wf1_score
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            break
    
    print(f'[INFO]: Early stopping after {epoch} epochs.')

    # load the best model
    model.load_state_dict(best_model)
    
    return model, best_loss, best_wf1, loss_history



if __name__ == '__main__':

    args = input_parse()

    # hyperparameter grid
    BATCH_SIZE_OPTIONS = [16, 32, 64]
    LEARNING_RATE_OPTIONS = [0.001, 0.01]
    DROPOUT_OPTIONS = [0.2, 0.3, 0.4]
    TWO_LAYERS = [True, False]
    hyperparameter_grid = list(product(BATCH_SIZE_OPTIONS, LEARNING_RATE_OPTIONS, DROPOUT_OPTIONS, TWO_LAYERS))
    
    K_FOLDS = 10
    PATIENCE = 5  # number of epochs to wait for improvement
    DELTA = 0.001  # minimum change in validation loss to qualify as an improvement

    # paths
    root = Path(__file__).parents[1]
    data_path = root / 'data' / 'processed' / f'{args.feature}_features_train.pkl'

    # load data
    with open(data_path, 'rb') as f:
        X_train, y_train = pickle.load(f)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # criterion
    criterion = nn.BCELoss(reduction='mean')

    # track the best hyperparameters
    best_hyperparameters = None
    best_avg_loss = None
    best_avg_wf1 = 0
    best_hyperparameters_model = None
    best_hyperparameters_loss_history = None

    for BATCH_SIZE, LEARNING_RATE, DROPOUT, TWO_LAYERS in hyperparameter_grid:
        print(f'---------------------------------\n[INFO]: Testing Hyperparameters - BATCH_SIZE: {BATCH_SIZE}, LEARNING_RATE: {LEARNING_RATE}, DROPOUT: {DROPOUT}, TWO_LAYERS: {TWO_LAYERS}')

        # k-fold cross validation model evaluation
        kfold = KFold(n_splits=K_FOLDS, shuffle=True)
        fold_performance = []

        # track the best model for current hyperparameters
        best_fold_wf1 = 0

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            print(f'---------------------------------\n[INFO]: Fold {fold+1}/{K_FOLDS}')

            # create dataloaders for training and validation
            train_subset = Subset(TensorDataset(X_train, y_train), train_idx)
            val_subset = Subset(TensorDataset(X_train, y_train), val_idx)
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=len(val_idx), shuffle=False)

            # model
            model = Model(dropout=DROPOUT, two_layers=TWO_LAYERS).to(device)

            # optimizer
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # train the model with early stopping
            model, val_loss, wf1_score, loss_history = early_stopping(train_loader, val_loader, model, criterion, optimizer, device, PATIENCE, DELTA)
            fold_performance.append((val_loss, wf1_score))
            print(f'[INFO]: Loss: {val_loss}, F1: {wf1_score}')

            # save the best model for the current hyperparameters
            if wf1_score > best_fold_wf1:
                best_fold_wf1 = wf1_score
                best_fold_model = model.state_dict()
                best_fold_loss_history = loss_history

        # calculate average performance across folds for the current hyperparameters
        avg_loss = np.mean([perf[0] for perf in fold_performance])
        avg_wf1 = np.mean([perf[1] for perf in fold_performance])

        # save the best hyperparameters and model if current hyperparameters are better
        if avg_wf1 > best_avg_wf1:
            best_avg_wf1 = avg_wf1
            best_avg_loss = avg_loss
            best_hyperparameters = (BATCH_SIZE, LEARNING_RATE, DROPOUT, TWO_LAYERS)
            best_hyperparameters_fold_performance = fold_performance
            best_hyperparameters_model = best_fold_model
            best_hyperparameters_loss_history = best_fold_loss_history

        print(f'---------------------------------\n[INFO]: Hyperparameters - BATCH_SIZE: {BATCH_SIZE}, LEARNING_RATE: {LEARNING_RATE}, DROPOUT: {DROPOUT}, TWO_LAYERS: {TWO_LAYERS}')
        print(f'[INFO]: Average Loss: {avg_loss}, Average Weighted F1: {avg_wf1}')

    # print best hyperparameters and their performance
    print(f'---------------------------------\n[INFO]: Best Hyperparameters - BATCH_SIZE: {best_hyperparameters[0]}, LEARNING_RATE: {best_hyperparameters[1]}, DROPOUT: {best_hyperparameters[2]}, TWO_LAYERS: {best_hyperparameters[3]}')
    print(f'[INFO]: Best Average Weighted F1: {best_avg_wf1}, Average Loss: {best_avg_loss}')

    # save the best model, performance and best training history
    model_dir = root / 'models' / f'nn_{args.feature}_f1_{round(best_avg_wf1, 4)}'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(best_hyperparameters_model, model_dir / f'nn_{args.feature}_model.pth')
    
    with open(model_dir / 'cross_val_performance.txt', 'w') as f:
        for fold, (loss, f1) in enumerate(best_hyperparameters_fold_performance):
            f.write(f'Fold {fold+1} | Loss: {loss} | Weighted F1: {f1}\n')
        f.write(f'\nAverage Loss: {best_avg_loss}, Average Weighted F1: {best_avg_wf1}\n')
        f.write(f'Best fold based on weighted F1: {np.argmin([perf[1] for perf in best_hyperparameters_fold_performance]) + 1}')
    
    best_hyperparameters_loss_history.to_csv(model_dir / 'best_loss_history.csv', index=False)

    # save best hyperparameters along with model architecture
    with open(model_dir / 'hyperparameters.txt', 'w') as f:
        f.write(f'Batch Size: {best_hyperparameters[0]}\nLearning Rate: {best_hyperparameters[1]}\nDropout: {best_hyperparameters[2]}\nTwo Layers: {best_hyperparameters[3]}\nK-Folds: {K_FOLDS}\nPatience: {PATIENCE}\nDelta: {DELTA}')
        f.write(f'\n\nModel Architecture:\n{model}')
        
    print(f'[DONE]: Best model, cross-validation performances, and training history of the best model saved at {model_dir}.')
