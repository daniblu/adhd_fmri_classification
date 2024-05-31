# adhd_fmri_classification
### Background
This repository contains the code for an exam project for the course [Data Science, prediction, and forecasting](https://kursuskatalog.au.dk/da/course/122493/Data-Science-Prediction-and-Forecasting) at Aarhus University. The repositroy allows for complete replication of the entire analysis. Note, the repository and the following manual assumes a POSIX system.

### Project abstract
Traditional procedures of diagnosing attention-deficit/hyperactivity disorder (ADHD) are associated with a range of challenges. Leaning against several previous studies, this paper explores the potential of using machine learning algorithms applied to fMRI data to enhance the diagnosis of ADHD. Utilizing the publicly available [ADHD-200 dataset](http://fcon_1000.projects.nitrc.org/indi/adhd200/#), three models were trained for the binary classification task of distinguishing ADHD subjects from healthy controls: a logistic regression fitted only to phenotypical data, a support vector machine, and a fully connected feed-forward neural network fitted to two different features extracted from fMRI time-courses, namely average signal value and eigenvector centrality. Logistic regression achieved the highest weighted F1 score (.61) and test accuracy (.63), but all models displayed underwhelming test performance. Despite unreliable models, feature importance was calculated and highlighted the left superior temporal gyrus as a highly discriminative brain area, in agreement with the literature. It is argued that the low performances of the models fitted to fMRI data can likely be traced to the choice of feature extraction method and noise in the data itself.

### Repository setup
Clone the repository and setup the environment by running
```bash
bash setup_env.sh
```

Before running any scripts, the environment should be activated:
```bash
source env/bin/activate
```

Should the user wish to experiment within the environment using a Python notebook, install an environment kernel by running
```bash
bash setup_nb_kernel.sh
```

### Pipeline
Make sure all necessary data is downloaded. See ``data/ADHD-200/README.md`` for guidance. With the terminal directory being the root of the repository, run the following scripts (in this order!) to prepare the data for the machine learning models:
```bash
python3 src/prep_pheno_data.py
python3 src/prep_neuro_data.py
```

The following scripts perform hyperparameter tuning using 10 fold cross-validation for each of the model. For the SVM and the neural net, the user must also specify which type of feature the model should be fitted to. The options are "average" or "centrality". For example:
```bash
python3 src/fit_logistic.py
python3 src/fit_svm.py -f average
python3 src/fit_neuralnet.py -f centrality
```
Each tuned model will be saved in its own folder in ``models``. The subfolder name will be ``[model type]_[feature type]_[average weighted f1 score]``. All outputs related to subsequent analysis of the model will be saved to this folder, including test evaluation overviews and plots.

Before evaluation of the FFNN models is possible, a baseline vector should be estimated for calculating integrated gradients of the true positive test predictions (integrated gradients is a method of determining feature importance). The user must specify the specific neural net model by referring to the model's folder name. For example:
```bash
python3 src/estimate_baseline -m nn_centrality_f1_0.573
```

A model of choice can be evaluated on the test data by running for example:
```bash
python3 src/evaluate.py -m nn_centrality_f1_0.573
```

Plots that indicate if there is any relation between model prediction patterns and ADHD index or ADHD subtype can be produced by running:
```bash
python3 src/plot_predictions.py -m nn_centrality_f1_0.573
```

Finally, the brain regions with the biggest contribution to true positive test predictions (as determined by integrated gradients) can be plotted by running the following. Note, this is only possible for neural net models for which the baseline has been computed:
```bash
python3 src/plot_top_features.py -m nn_centrality_f1_0.573
```

