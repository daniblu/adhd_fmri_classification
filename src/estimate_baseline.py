'''
This script estimates the baseline input for a custom neural network by optimizing the input vector to produce a target output of 0. 
This baseline vector is used for calculating integrated gradients in the evaluation script.
'''

print('[INFO]: Importing libraries.')
from pathlib import Path
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from fit_neuralnet import Model # check that the model class defined in fit_neuralnet.py matches the architecture of the model being loaded here



def input_parse():
    '''
    For parsing terminal arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-m", type=str, required=True, help="Name of directory containing model to load.")
    args = parser.parse_args()

    return args



def load_model(model_path: Path):
    '''
    Load model from directory.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    model_file = list(model_path.glob('*.pth'))[0]
    model.load_state_dict(torch.load(model_file, map_location=device))

    return model



if __name__ == '__main__':
    
    args = input_parse()
    
    root = Path(__file__).parents[1]
    model_path = root / 'models' / args.model_dir

    # Load the model
    model = load_model(model_path)

    # set the network to evaluation mode (to avoid dropout)
    model.eval()

    # target output
    target = torch.tensor([0.0])

    # initialize a random input vector, note requires_grad=True
    input_vector = torch.randn(111, requires_grad=True)

    # define the loss function
    loss_fn = nn.L1Loss(reduction='none')

    # define the optimizer to optimize the input vector
    optimizer = optim.Adam([input_vector], lr=0.1)

    # optimization loop
    step = 0
    print('---------------------------------\n[INFO]: Starting optimization.')
    while True:
        optimizer.zero_grad()
        output = model(input_vector)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        # check if the loss is below the threshold
        if loss.item() < 0.00001:
            break
        
        # print progress every 100 steps
        if step % 10 == 0:
            print(f'[INFO]: Step {step} | Output: {output.item()}')
        step += 1

    # save final input vector that should produce an output close to 0, save as txt
    input_vector = input_vector.detach().numpy()
    baseline_path = model_path / 'baseline.pkl'
    with open(baseline_path, 'wb') as f:
        pickle.dump(input_vector, f)
    
    print(f'[DONE]: Model output for optimized input: {output.item()}.\nBaseline saved to {baseline_path}.')
