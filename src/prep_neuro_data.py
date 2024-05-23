'''
This script prepares the training and testing data of brain parcel centrality features.
'''
print("[INFO]: Importing libraries.")
from pathlib import Path
import pandas as pd
import networkx as nx
import torch
import pickle
from tqdm import tqdm
from typing import Dict



def load_time_courses(path: Path) -> pd.DataFrame:
    '''
    Loads the 1D file containing ADHD-200 time courses at given path.
    Reformats the data as a data frame with parcels as columns and time steps as rows.
    '''

    with open(path) as f:
        data = f.readlines()

    # seperate time steps for each parcel
    data = [line.split("\t") for line in data]

    # remove spaces and new lines
    for i, list in enumerate(data):
        data[i] = [line.strip() for line in list]

    # remove header information
    data = [list[2:] for list in data]

    # create dictionary with parcel as key and time series as values
    data_dict = {data[0][i]: [list[i] for list in data[1:]] for i in range(len(data[0]))}

    # create data frame
    data_df = pd.DataFrame(data_dict)

    # convert to float
    data_df = data_df.astype(float)

    return data_df



def compute_eigen_centrality(df: pd.DataFrame) -> Dict:
    '''
    Computes the correlation matrix from the input df. 
    Then, eigencentrality of each parcel is computed and saved as a dictionary.
    '''

    # calculate correlation matrix
    corr_matrix = df.corr()

    # convert to absolute values
    corr_matrix = corr_matrix.abs()

    # create graph
    G = nx.from_pandas_adjacency(corr_matrix, create_using=nx.Graph)

    # compute eigenvector centrality
    eigenvector_centrality = nx.eigenvector_centrality(G, weight="weight")
    
    return eigenvector_centrality



def compute_tc_average(df: pd.DataFrame) -> Dict:
    '''
    Computes the average value of each parcel's time course.
    '''
    
    # compute column-wise average
    average = df.mean(axis=0)

    return average



def normalize_data(data: torch.tensor) -> torch.tensor:
    '''
    Normalizes the input data.
    '''
    return (data - data.mean(axis=0)) / data.std(axis=0)



if __name__ == '__main__':

    # paths
    root = Path(__file__).parents[1]
    tc_train_path = root / 'data' / 'ADHD-200' / 'ADHD200_HO_TCs_filtfix'
    tc_test_path = root / 'data' / 'ADHD-200' / 'ADHD200_HO_TCs_TestRelease'
    pheno_path = root / 'data' / 'processed' / 'pheno_data_train_test.csv'

    # train and test dictionaries to store eigenvector centralities for each subject across all sites
    eigen_centralities = [{}, {}]

    # train and test dictionaries to store time course averages for each subject across all sites
    time_course_averages = [{}, {}]

    # list for missing subject data
    missing_subjects = []

    # extract features for training data, then testing data
    set_paths = [tc_train_path, tc_test_path]
    for i, tc_path in enumerate(set_paths):

        set = 'train' if i == 0 else 'test'

        # list all site paths
        sites = [folder for folder in tc_path.iterdir() if folder.is_dir() and folder.name != 'templates']

        # loop over sites
        for site in sites:

            # list all subject paths
            subject_paths = [folder for folder in site.iterdir() if folder.is_dir()]

            # loop over subjects
            for subject_path in tqdm(subject_paths, desc=f'[INFO] Processing {set} subjects at {site.name}'):

                # get subject id
                subject = subject_path.name
                
                # get the file name of the filtered time courses
                try:
                    file_name = [file for file in subject_path.iterdir() if file.name.startswith('sf')][0]
                except IndexError:
                    missing_subjects.append(f'No filtered time course file | Site: {site.name}, Subject: {subject}')
                    continue

                # load time courses
                tc_path = subject_path / file_name
                df = load_time_courses(tc_path)

                # compute eigenvector centrality
                eigenvector = compute_eigen_centrality(df)
                eigenvector = list(eigenvector.values())

                # store eigenvector centrality in dictionary
                eigen_centralities[i][subject] = eigenvector

                # compute time course averages
                average = compute_tc_average(df)

                # store time course averages in dictionary
                time_course_averages[i][subject] = average

    # load phenotypic data
    pheno_data = pd.read_csv(pheno_path, dtype={"ScanDir ID": str})

    # collect all keys from eigen_centralities
    tc_subjects_train = list( eigen_centralities[0].keys() )
    tc_subjects_test = list( eigen_centralities[1].keys() )
    tc_subjects_all = tc_subjects_train + tc_subjects_test

    # create training and testing data of centrality features (some test subjects with tc data are not in pheno_data because their ADHD status is missing)
    X_train_centrality = torch.tensor( [eigen_centralities[0][subject] for subject in tc_subjects_train], dtype=torch.float32 )
    X_test_centrality = torch.tensor( [eigen_centralities[1][subject] for subject in tc_subjects_test if subject in pheno_data['ScanDir ID'].values], dtype=torch.float32 )

    # normalize centrality features
    X_train_centrality = normalize_data(X_train_centrality)
    X_test_centrality = normalize_data(X_test_centrality)

    # create training and testing data of time course averages (some test subjects with tc data are not in pheno_data because their ADHD status is missing)
    X_train_averages = torch.tensor( [time_course_averages[0][subject] for subject in tc_subjects_train], dtype=torch.float32 )
    X_test_averages = torch.tensor( [time_course_averages[1][subject] for subject in tc_subjects_test if subject in pheno_data['ScanDir ID'].values], dtype=torch.float32 )

    # normalize average features
    X_train_averages = normalize_data(X_train_averages)
    X_test_averages = normalize_data(X_test_averages)

    # create training and testing labels (some test subjects with tc data are not in pheno_data because their ADHD status is missing. This time those subjects are recorded in missing_subjects)
    y_train = torch.tensor( [pheno_data['ADHD'][pheno_data['ScanDir ID']==subject].values[0] for subject in tc_subjects_train]).reshape(-1, 1).float()
    
    y_test = []
    for subject in tc_subjects_test:
        try:
            y_test.append( pheno_data['ADHD'][pheno_data['ScanDir ID']==subject].values[0] )
        except IndexError:
            missing_subjects.append(f'No ADHD status | Subject: {subject}')
    y_test = torch.tensor( y_test ).reshape(-1, 1).float()

    # correct tc_subjects_test to only include subjects with pheno data
    tc_subjects_test = [subject for subject in tc_subjects_test if subject in pheno_data['ScanDir ID'].values]

    # save data
    with open(root / 'data' / 'processed' / 'centrality_features_train.pkl', 'wb') as f:
        pickle.dump((X_train_centrality, y_train), f)
    
    with open(root / 'data' / 'processed' / 'centrality_features_test.pkl', 'wb') as f:
        pickle.dump((X_test_centrality, y_test), f)
    
    with open(root / 'data' / 'processed' / 'average_features_train.pkl', 'wb') as f:
        pickle.dump((X_train_averages, y_train), f)

    with open(root / 'data' / 'processed' / 'average_features_test.pkl', 'wb') as f:
        pickle.dump((X_test_averages, y_test), f)

    # save test subject order
    with open(root / 'data' / 'processed' / 'test_subject_order_neuro.txt', 'w') as f:
        for subject in tc_subjects_test:
            f.write(f'{subject}\n')
    
    # save missing subjects
    with open(root / 'data' / 'processed' / 'missing_subjects.txt', 'w') as f:
        for subject in missing_subjects:
            f.write(f'{subject}\n')

    print(f'[DONE]: Features extracted and saved to {root / "data" / "processed"}\nShape of X_train: {X_train_centrality.shape}\nShape of X_test: {X_test_centrality.shape}\nShape of y_train: {y_train.shape}\nShape of y_test: {y_test.shape}')

