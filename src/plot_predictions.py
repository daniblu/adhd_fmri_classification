'''
This script loads the predictions of a custom model and creates two plots:
- A violin plot showing distribution of ADHD index within each accuracy category (True Positive, False Positive, True, Negative, False Negative)
- A horizontal stacked bar plot for each type of sub-diagnosis coloring by whether the subjects were correctly and incorrectly predicted.
'''

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def input_parse():
    '''
    For parsing terminal arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-m", type=str, required=True, help="Name of directory containing model to plot.")
    args = parser.parse_args()

    return args



if __name__ == '__main__':

    args = input_parse()

    # paths
    root = Path(__file__).parents[1]
    predictions_path = root / 'models' / args.model_dir / 'predictions.csv'
    pheno_path = root / 'data' / 'processed' / 'pheno_data_train_test.csv'

    # load data
    predictions = pd.read_csv(predictions_path, dtype={'subject_id': str})
    pheno_data = pd.read_csv(pheno_path, dtype={'ScanDir ID': str})

    # rename ScandDir ID to subject_id
    pheno_data.rename(columns={'ScanDir ID': 'subject_id'}, inplace=True)

    # merge data on subject_id
    predictions = predictions.merge(pheno_data[['subject_id', 'ADHD Index', 'DX']], on='subject_id')

    # create accuracy categories column
    predictions['pred_category'] = 'False Negative'
    predictions.loc[(predictions['true_adhd'] == 1) & (predictions['pred_adhd'] == 1), 'pred_category'] = 'True Positive'
    predictions.loc[(predictions['true_adhd'] == 0) & (predictions['pred_adhd'] == 1), 'pred_category'] = 'False Positive'
    predictions.loc[(predictions['true_adhd'] == 0) & (predictions['pred_adhd'] == 0), 'pred_category'] = 'True Negative'

    # create diagnosis categories column
    predictions['diagnosis'] = 'Neurotypical'
    predictions.loc[predictions['DX'] == 1, 'diagnosis'] = 'ADHD-Combined'
    predictions.loc[predictions['DX'] == 2, 'diagnosis'] = 'ADHD-Hyperactive/Impulsive'
    predictions.loc[predictions['DX'] == 3, 'diagnosis'] = 'ADHD-Inattentive'

    # create string column indicating if prediction was correct or incorrect
    predictions['correct'] = 'Incorrect'
    predictions.loc[predictions['true_adhd'] == predictions['pred_adhd'], 'correct'] = 'Correct'

    # save predictions for inspection
    predictions.to_csv(root / 'models' / args.model_dir / 'predictions_with_data.csv', index=False)

    # prepare data for violin plot
    predictions_violin = predictions[['ADHD Index', 'pred_category']].dropna()
    predictions_violin = predictions_violin[predictions_violin['ADHD Index'] > 0]

    # plot violin plot, increase x axis tick labels
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='pred_category', y='ADHD Index', data=predictions_violin, 
                   order=['True Positive', 'False Negative', 'True Negative', 'False Positive'],
                   fill=False,
                   linewidth=2,
                   inner='point', 
                   color='black')
    plt.xlabel('')
    plt.ylabel('ADHD Index', fontsize=16)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=16)
    ax_size = plt.gca()
    plt.savefig(root / 'models' / args.model_dir / 'adhd_index_distribution.png')
    plt.close()

    # prepare data for stacked bar plot
    predictions_bar = predictions.groupby(['diagnosis', 'correct']).size().reset_index().pivot(columns='correct', index='diagnosis', values=0)
    order = ['ADHD-Hyperactive/Impulsive', 'ADHD-Inattentive', 'ADHD-Combined', 'Neurotypical']
    predictions_bar = predictions_bar.reindex(order)

    # plot stacked bar plot
    predictions_bar.plot(figsize=(8, 4),
                         kind='barh', 
                         stacked=True, 
                         color=['#bfbcbb', '#7a7978'],
                         edgecolor='black')
    plt.legend(title='', fontsize=14, loc='lower right')
    plt.xlabel('Number of test samples', fontsize=14)
    plt.ylabel('')
    plt.yticks(fontsize=14)
    plt.tick_params(left=False)
    plt.tight_layout()
    plt.savefig(root / 'models' / args.model_dir / 'diagnosis_proportion.png')


    print(f'[DONE]: Plots saved in {root / "models" / args.model_dir}.')