# OD Transportability Code
Note: My apologies for the lack of version history in this repo. The actual code was developed in a (much messier) private repo including my notes along the way and tons of extraneous files created during the development process which I have not made public. I can show the original repo on request for authorship verification purposes.

## Credits
The basic inspiration for this code (and this paper at large) comes from Melika Farahani (UBC MS 2022) and her original Master's Thesis work on transportability. This project expanded the research significantly, and as a result, the code in this repo was completely re-written from scratch by me, Ziven Anderson. However, some methods and underlying design decisions were inspired by Melika's original code: https://github.com/melikaf/TranspotableModels

## Packages & Dependencies
All code in this repo is either in Python files or Python Jupyter notebooks. I ran everything on a Python 3.10 environment on my Windows machine. The packages used are as follows:
* xgboost
* scikit-learn
* pandas
* cupy (CUDA-acceleration-compatible drop-in replacement for NumPy)

## Files and their Purpose

### Datasets and Results
Datasets to be run through the models are stored as csv files in the Datasets directory. The dataset citations in the paper refer to the actual sources where the original datasets were retrieved from, while the datasets in this directory are the exact processed versions that I actually ran to generate the paper results. The respective csv results from running models on the datasets are stored in the Results directory (git-ignored to avoid repo clutter).

### `setup.py`
This file contains all the globally necessary variables and definitions to run the models on a given dataset. This includes things like the filepath where the dataset can be found, default values of the features in the dataset, and parameters controlling the iterations that will be run. See details on all options in the code comments. All datasets have definitions in the `setup.py` file. Uncomment a given dataset's setup code (and comment out the rest of the file) to work with that dataset.

### `dataset_bin.ipynb` & `dataset_properties.ipynb`
These notebooks contain some basic code I used to process datasets (notably, bin continuous values into categorical quartiles) as well as to get information such as the infogain of the individual features of a dataset (which is used in the paper).

### `simple_models_single_split.py`
Running this file generates the csv data necessary to make figure X.1 from the paper (for any dataset). It only runs on the first source-target zone given, and it provides the loss on a feature-by-feature basis from training single-feature models on each of them and comparing CP and OD outputs. It uses laplace smoothing on top of a basic count-based MLE estimate.

### `simple_models.py`
Running this file generates the csv data necessary to make figure X.2 from the paper (for any dataset). It iterates through all given source-target population zones given and provides one average loss from all single-feature models for each population pair. Again, using laplace smoothing on top of a basic count-based MLE estimate. For figure 3, choose a pre-defined random set of columns, for figure 4, use dataset_properties.ipynb to get the most important columns to split by (take all with importance within 25% of max importance)

### `soph_models.py`
Running this file generates the csv data necessary to make figures X.3 and X.4 from the paper (for any dataset). It runs several times with different numbers of features (as directed in setup.py). During each of these runs, it iterates through each setup.py-defined source-target population pair 20 times, training an XGB model with num_feats features which are dynamically randomly sampled from the available features in the dataset. The ratio between OD log loss and CP log loss is the ultimate saved value, and is split based on number of features. Raw losses also saved for reference (though these are not explicitly presented in the paper)
