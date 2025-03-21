# Spiro-CLF (Spirogram-based Contrastive Learning Framework)

This repository contains the source code for the manuscript "Deep Learning of Suboptimal Spirometry to Predict Respiratory Outcomes and Mortality"

![Image](https://github.com/davinhill/Spiro-CLF/blob/main/Figures/Figure_SpiroCLF.png?raw=true)


## Requirements
The repository requirements are listed in the requirements.txt file. We recommend using a virtual environment.
```
$ conda create --name spiroclf --file requirements.txt
```

## Spiro-CLF Training

The Spiro-CLF model is trained using the train_spiroclf.py script. We recommend using a config file to select model parameters, which can be passed to the training script using '-c'. The model parameters used in the manuscript are provided in config/spiroclf_config.yaml.
We recommend using Weights and Biases to track model progress, which can set using the --wandb_logging flag.

```
$ python train_spiroclf.py -c config/spiroclf_config.yaml
```

The main options to adjust are data_path, and id_path.
* **data_path** is the path to the data file
* **id_path** is the path to a directory containing row indices corresponding to the train/test/validation splits. This directory should contain 'train_id_full.npy', 'train_id_full.npy', 'train_id_full.npy', respectively, which are numpy arrays of row indices. Please note that these indices do not correspond to any participant identifiers and are only used for reproducibility purposes.

The model architecture is contained in the **model_spiroclf.py** script.

The Pytorch dataset loaders are contained in the **load_data.py** script. The **Spiro_ukbb** class corresponds to the UK Biobank dataset; the **Spiro_copdgene** corresponds to the COPDGene dataset.



## Experiments

The Experiments directory contains the code that reproduces the results in the manuscript.

### Section 3.1: Lung Function Impairment Prediction with Spiro-CLF Representations

The **a_LP_ratio/LP_ratio.py** and **b_LP_FEV1pp/FEV1pp_binary.py** scripts reproduce the linear probe testing in Section 3.1 of the manuscript. Both scripts generate the Spiro-CLF representations (or use the pre-calculated representations), then train a linear model to predict the binary outcomes of FEV1/FVC ratio < 0.7 or FEV1pp < 0.8.

### Section 3.2: All-Cause Mortality Prediction with Spiro-CLF Features
The **c_mortality/mortality.py** script takes the pre-calculated Spiro-CLF representations, then trains a Cox Regression model on mortality. The **c_mortality/mortality_competingmethods.py** script calculates traditional alternative spirometry metrics for each participant, then Cox Regression model using the alternative metrics as predictors. These models are used for comparing against the Spiro-CLF representations.


### Section 3.3: Phenotype Prediction
The **d_phewas/LP_phewas.py** script takes the pre-calculated Spiro-CLF representations, then trains a linear model to predict various respiratory phenotypes. The specific model type for each phenotype is specified in line 593 of the script.
