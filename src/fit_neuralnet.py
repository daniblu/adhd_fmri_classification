'''
This scripts fits a neural network to the training data and saves the model.
The type of features used as training data are either eigenvector centralities or time course averages, which can be selected from the terminal. 
'''

print('[INFO]: Importing libraries.')
from pathlib import Path
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm



def input_parse():
    '''
    For parsing terminal arguments.
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument("--feature", "-f", choices=['centrality', 'average'], required=True, help="Type of feature to use for training.")
    args = parser.parse_args()

    return args



def create_dataloader(X, y, batch_size=16):
    '''
    Convert torch tensor data to torch Dataloader
    '''
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(111, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
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



if __name__=='__main__':

    args = input_parse()
 
    # hyperparameters
    EPOCHS = 70
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001

    # paths
    root = Path(__file__).parents[1]
    data_path = root / 'data' / 'processed' / f'{args.feature}_features_train.pkl'

    # load data
    with open(data_path, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    # create dataloader
    train_loader = create_dataloader(X_train, y_train, batch_size=BATCH_SIZE)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = Model().to(device)

    # criterion and optimizer
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train model
    training_history = []

    for epoch in tqdm(range(EPOCHS), desc='[INFO]: Training epoch'):
        epoch_performance = train_model(model, train_loader, criterion, optimizer, device)
        # save training history
        training_history.append(f'Epoch: {epoch+1} | Loss: {epoch_performance[0]} | Accuracy: {epoch_performance[1]}')
    
    # save model and loss history
    model_dir = root / 'models' / f'{args.feature}_nn_e{EPOCHS}_bs{BATCH_SIZE}_lr{LEARNING_RATE}'
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / f'{args.feature}_nn_model.pth')
    with open(model_dir / 'training_history.txt', 'w') as f:
        for item in training_history:
            f.write(f'{item}\n')
        

    print(f'[DONE]: Model and training history saved at {model_dir}.')


