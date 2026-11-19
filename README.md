## Language
*   [English](README.md)
*   [中文](README_CN.md)

## Introduction
*   This repository contains all the code for constructing the training set and training GEoS, providing a complete workflow for model training.
*   Because the test set includes proprietary commercial data, it cannot be publicly released. Therefore, only the calculation results from each model are provided.
*   Additionally, we provide the fitted PC-SAFT parameters along with their fitting details, such as the mean squared error (MSE) and fitting time, for testing and reference.

## File Description
*   The `Feature` directory contains the feature selection results and the code used to determine the final feature set.
    *   `RHO-Boruta.csv` contains the feature selection results.
    *   `EM.ipynb` contains the code for the Expectation-Maximization (EM) algorithm used in feature selection.
*   The `Database` directory contains the code for constructing the training set.
    *   `features.xlsx` contains the feature information.
    *   `feature_statistics.xlsx` contains the mean and standard deviation of the training features.
    *   `build.ipynb` is the script for constructing the training set.
    *   Running `build.ipynb` generates the `database_new_max.pt` and `database_new_max_val.pt` files, which serve as the final training set and validation set, respectively (note that half of the validation set is incorporated into the training set during actual training).
*   The `Test` directory contains the calculation results on the test set.
    *   `pcsaft_regression_results.csv` contains the fitted PC-SAFT parameters.
*   The `Training` directory contains the code for model training.
    *   `train.py` is the script for model training.
    *   `trainall.sh` is the shell script to train all models.
    *   `fine.py` is the script for model fine-tuning.
    *   `fineall.sh` is the shell script to fine-tune all models.
    *   `model.py` defines the model architecture.
    *   The `img` directory stores the loss curves generated during training.
    *   The `log` directory stores the training logs.
    *   The `model` directory stores the trained model files.

## Usage Guide
1. Clone the repository
   ```bash
   git clone https://github.com/A-Normal-User/GEoS.git
   cd GEoS
   ```

2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```

3. View feature selection results
   ```bash
   cd Feature
   jupyter notebook EM.ipynb
   cd ..
   ```

4. Construct the training set
   ```bash
   cd Database
   jupyter notebook build.ipynb
   cd ..
   ```

5. Train the pre-trained models
    ```bash
    cd Training
    bash trainall.sh
    cd ..
    ```

6. Fine-tune the models after pre-training
    ```bash
    cd Training
    bash fineall.sh
    cd ..
    ```

### Notes
*   `build.ipynb` matches the code used to construct the training set in our paper. However, variations in hardware environments may cause minor differences in the generated files. For reference, the MD5 checksums of our `.pt` files are provided below:
    *   MD5 checksum of `database_new_max.pt`: `dec7b213c5413ee6e8e703adfe282568`
    *   MD5 checksum of `database_new_max_val.pt`: `f6710f57386d59e67091f61eab6c5094`

*   Model training is conducted on `Ubuntu 22.04.4 LTS` using `Python 3.10.12` and PyTorch `2.5.1+cu121`. Two `NVIDIA GeForce RTX 3090` GPUs are utilized: models with an even `index` are trained on the first GPU, while those with an odd `index` are trained on the second GPU.
    *   Because training results can vary slightly across different hardware environments, we provide our trained models in the `Training/model` directory.

## File Structure
```
|   README.md
|   README_CN.md
|
+---Feature
|       EM.ipynb
|       RHO-Boruta.csv
|
+---Database
|       build.ipynb
|       features.xlsx
|       feature_statistics.xlsx
|
+---Test
|       pcsaft_regression_results.csv
|
\---Training
    |   fine.py
    |   fineall.sh
    |   model.py
    |   train.py
    |   trainall.sh
    |
    +---img
    +---log
    \---model
            model_DGM_fine_1131796.pth
            model_DGM_fine_1211200801.pth
            model_DGM_fine_157564161.pth
            model_DGM_fine_18094361.pth
            model_DGM_fine_218105633.pth
            model_DGM_fine_26317076.pth
            model_DGM_fine_26318864.pth
            model_DGM_fine_448585456.pth
            model_DGM_fine_539035672.pth
            model_DGM_fine_609641183.pth
            model_max_1131796.pth
            model_max_1211200801.pth
            model_max_157564161.pth
            model_max_18094361.pth
            model_max_218105633.pth
            model_max_26317076.pth
            model_max_26318864.pth
            model_max_448585456.pth
            model_max_539035672.pth
            model_max_609641183.pth
```