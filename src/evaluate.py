'''
This script loads a model of choice and evaluates it on the test set. It returns a csv file with four columns: subject ID, true ADHD status, predicted ADHD status, and an indicator of whether the prediction was correct.
'''

from pathlib import Path
import pickle
import pandas as pd
import argparse
import torch
from sklearn.metrics import classification_report 
from captum.attr import IntegratedGradients
from typing import Dict 
from fit_neuralnet import Model



def input_parse():
    '''
    For parsing terminal arguments.
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-m", type=str, required=True, help="Name of directory containing model to be evaluated.")
    args = parser.parse_args()

    return args



def load_model(model_path: Path):
    '''
    Load model from directory.
    '''

    # check model type by looking at the model filename (contains either "nn", "svm", or "logistic")
    splitted_path = model_path.stem.split('_')
    model_info = {'type': splitted_path[0], 'feature': splitted_path[1]}

    if model_info['type'] == 'nn':

        with open(model_path / 'hyperparameters.txt', 'r') as f:
            hyperparameters = f.read().splitlines()
        TWO_LAYERS = hyperparameters[3].split(': ')[1] == 'True'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model(two_layers=TWO_LAYERS).to(device)
        model_file = list(model_path.glob('*.pth'))[0]
        model.load_state_dict(torch.load(model_file, map_location=device))

    else:
        with open(model_path / 'best_model.pkl', 'rb') as f:
            model = pickle.load(f)

    return model_info, model



def load_data(path: Path, model_info: Dict):
    '''
    According to the model type:
    - Loads test_subject_order_neuro.txt or test_subject_order_pheno.txt into a list.
    - Loads test features and labels.
    '''

    if model_info['type'] == 'logistic':
        with open(path / 'test_subject_order_pheno.txt', 'r') as f:
            subject_ids = f.read().splitlines()

        with open(path / 'pheno_test.pkl', 'rb') as f:
            X_test, y_test = pickle.load(f)
            y_test = y_test.tolist()
    
    else:
        with open(path / 'test_subject_order_neuro.txt', 'r') as f:
            subject_ids = f.read().splitlines()

        with open(path / f'{model_info["feature"]}_features_test.pkl', 'rb') as f:
            X_test, y_test = pickle.load(f)
            y_test = y_test.flatten().tolist()
    
    return subject_ids, X_test, y_test



def create_summary_dict(classification_report: Dict):
    '''
    Create a dictionary with precision, recall, f1, and accuracy.
    '''

    summary_dict = {'precision': {'0': classification_report['0.0']['precision'],
                                  '1': classification_report['1.0']['precision'],
                                  'weighted': classification_report['weighted avg']['precision']},
                    'recall': {'0': classification_report['0.0']['recall'],
                               '1': classification_report['1.0']['recall'],
                               'weighted': classification_report['weighted avg']['recall']}, 
                    'f1': {'0': classification_report['0.0']['f1-score'],
                           '1': classification_report['1.0']['f1-score'],
                           'weighted': classification_report['weighted avg']['f1-score']}, 
                    'accuracy': classification_report['accuracy']}

    return summary_dict



def compute_ig(model, model_path, X_test):
    '''
    Compute integrated gradients for a given model and input.
    '''

    # load baseline
    with open(model_path / 'baseline.pkl', 'rb') as f:
        baseline = pickle.load(f)
    baseline = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0)

    # compute integrated gradients and sum across all features
    ig = IntegratedGradients(model)
    ig_attributions = ig.attribute(X_test, target=0, baselines=baseline, n_steps=150)
    ig_attributions = ig_attributions.cpu().numpy()
    ig_attributions = ig_attributions.sum(axis=0)

    return ig_attributions



def evaluate_nn(model, model_path, X_test, y_test):
    '''
    Evaluate neural network model on test set.
    '''

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        
    y_pred = (y_pred > 0.5).float()
    y_pred = y_pred.cpu().numpy().flatten().tolist()

    report = classification_report(y_test, y_pred, output_dict=True)

    summary_dict = create_summary_dict(report)

    # get indices of where prediction and true label are both 1.0
    correct_indices = [i for i, (pred, true) in enumerate(zip(y_pred, y_test)) if pred == true and true == 1.0]
    X_test_correct = X_test[correct_indices]

    # compute integrated gradients
    ig_attributions = compute_ig(model, model_path, X_test_correct)


    return y_pred, summary_dict, ig_attributions



def evaluate_sklearn(model, X_test, y_test):
    '''
    Evaluate sklearn model on test set.
    '''

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    
    summary_dict = create_summary_dict(report)

    return y_pred, summary_dict



def create_pred_df(y_pred, y_test, subject_ids):
    '''
    Create dataframe with predictions.
    '''

    pred_df = pd.DataFrame({'subject_id': subject_ids, 'true_adhd': y_test, 'pred_adhd': y_pred})
    pred_df['subject_id'] = pred_df['subject_id'].astype(str)

    return pred_df



if __name__ == '__main__':

    args = input_parse()

    # paths
    root = Path(__file__).parents[1]
    model_path = root / 'models' / args.model_dir
    data_path = root / 'data' / 'processed'

    # load model
    model_info, model = load_model(model_path)

    # load data
    subject_ids, X_test, y_test = load_data(data_path, model_info)

    # evaluate model
    print(f'[INFO]: Evaluating {model_info["type"]} model.')
    if model_info['type'] == 'nn':
        y_pred, summary_dict, ig_attributions = evaluate_nn(model, model_path, X_test, y_test)

        # save integrated gradients (a numpy array) to txt file
        with open(model_path / 'ig_attributions.txt', 'w') as f:
            for i in ig_attributions:
                f.write(f'{i}\n')
    
    else:
        y_pred, summary_dict = evaluate_sklearn(model, X_test, y_test)

    # create dataframe with predictions
    pred_df = create_pred_df(y_pred, y_test, subject_ids)

    # save predictions
    pred_df.to_csv(model_path / 'predictions.csv', index=False)

    # save evaluation summary
    with open(model_path / 'test_eval_summary.txt', 'w') as f:
        f.write(str(summary_dict))
    

    print(f'[DONE]: Predictions and evaluation summary saved to {model_path}')