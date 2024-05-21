'''
This script fits a neural network to the training data using 10-fold cross-validation with early stopping.
The type of features used as training data are either eigenvector centralities or time course averages, which can be selected from the terminal.
'''

print('[INFO]: Importing libraries.')
from pathlib import Path
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold
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
    def __init__(self):
        super(Model, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(111, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logit = self.linear_relu_stack(x)
        return logit



def train_model(model, train_loader, criterion, optimizer, device):
    '''
    Train model for one epoch. Returns the average loss across batches and the average accuracy.
    '''
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += ((outputs > 0.5) == labels).sum().item() / len(labels)
    
    avg_loss = epoch_loss / len(train_loader)
    avg_acc = epoch_acc / len(train_loader)

    return avg_loss, avg_acc



def evaluate_model(model, val_loader, criterion, device):
    '''
    Evaluate model on validation data. Returns the average loss and accuracy.
    '''
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss = loss.item()
            val_acc = ((outputs > 0.5) == labels).sum().item() / len(labels)

    return val_loss, val_acc



def early_stopping(train_loader, val_loader, model, criterion, optimizer, device, patience, delta):
    '''
    Implements early stopping during training. Returns the best model, based on validation loss, along with its training and validation loss and accuracy. 
    '''
    best_loss = float('inf')
    best_model = None
    counter = 0
    training_history = pd.DataFrame(columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    for epoch in tqdm(range(1, 101), desc='Training model'):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # log training history
        training_history.loc[len(training_history.index)] = [train_loss, train_acc, val_loss, val_acc] 

        # check if validation loss improved
        if val_loss < best_loss - delta:
            best_loss = val_loss
            best_acc = val_acc
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            break
    
    print(f'[INFO]: Early stopping after {epoch} epochs.')

    # load the best model
    model.load_state_dict(best_model)
    
    return model, best_loss, best_acc, training_history



if __name__ == '__main__':

    args = input_parse()
 
    # hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
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

    # k-fold cross validation model evaluation
    kfold = KFold(n_splits=K_FOLDS, shuffle=True)
    fold_performance = []

    # track the best model
    best_fold_loss = float('inf')
    best_fold_acc = None
    best_fold_model = None
    best_fold_index = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f'---------------------------------\n[INFO]: Fold {fold+1}/{K_FOLDS}')

        # create dataloaders for training and validation
        train_subset = Subset(TensorDataset(X_train, y_train), train_idx)
        val_subset = Subset(TensorDataset(X_train, y_train), val_idx)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=len(val_idx), shuffle=False)

        # model
        model = Model().to(device)

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # train the model with early stopping
        model, val_loss, val_acc, training_history = early_stopping(train_loader, val_loader, model, criterion, optimizer, device, PATIENCE, DELTA)
        fold_performance.append((val_loss, val_acc))
        print(f'[INFO]: Loss: {val_loss}, Accuracy: {val_acc}')

        # save the best model
        if val_loss < best_fold_loss:
            best_fold_loss = val_loss
            best_fold_acc = val_acc
            best_fold_model = model.state_dict()
            best_fold_training_history = training_history
            best_fold_index = fold + 1

    # calculate average performance across folds
    avg_loss = np.mean([perf[0] for perf in fold_performance])
    avg_acc = np.mean([perf[1] for perf in fold_performance])

    # print average performance
    print(f'---------------------------------\n[INFO]: Average Loss: {avg_loss}, Average Accuracy: {avg_acc}')

    # save the best model, performance and best training history
    model_dir = root / 'models' / f'{args.feature}_nn_acc{round(avg_acc, 4)}'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(best_fold_model, model_dir / f'{args.feature}_nn_model.pth')
    
    with open(model_dir / 'cross_val_performance.txt', 'w') as f:
        for fold, (loss, acc) in enumerate(fold_performance):
            f.write(f'Fold {fold+1} | Loss: {loss} | Accuracy: {acc}\n')
        f.write(f'\nAverage Loss: {avg_loss}, Average Accuracy: {avg_acc}\n')
        f.write(f'Best Fold: {best_fold_index}')
    
    best_fold_training_history.to_csv(model_dir / 'best_training_history.csv', index=False)

    # save hyperparameters along with model architecture
    with open(model_dir / 'hyperparameters.txt', 'w') as f:
        f.write(f'Batch Size: {BATCH_SIZE}\nLearning Rate: {LEARNING_RATE}\nK-Folds: {K_FOLDS}\nPatience: {PATIENCE}\nDelta: {DELTA}')
        f.write(f'\n\nModel Architecture:\n{model}')
        
    print(f'[DONE]: Best model, cross-validation performances, and training history of the best model saved at {model_dir}.')
