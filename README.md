# 3DCNN_BMI
This repository includes the code used to run the 3D-CNN in the paper, "Deep Learning using 3D Convolutional Neural Network Analysis of Structural MRI Massively Improves Prediction Accuracy of Body Mass Index". The 3D-CNN was developed for prediction of BMI of individuals from the Human Connectome Project (https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release). 

The code assumes that the following files are in the same directory: Cleaning_data_bmi.py, CNN_split.py, Run_CNN_Fold.py, Run_Final_CNN_Fold.py, Run_Lockbox_CNN.py, CNN_functions.py, utils_tune.py, and models_tune.py. The utils_tune.py and models_tune.py code was adapted from that of Abrol et al. (2021).

The python files were run in the following order:
1. Cleaning_data_bmi.py (to split into train/test sets - saves to 'ssd' folder)
2. CNN_split.py (to split train set into nested folds - saves to 'ssd/cv' folder)
3. Run_CNN_Fold.py (to conduct hyperparameter tuning for a given fold)
4. Run_Final_CNN_Fold.py (to fit CNN to fold with optimal hyperparameters chosen in previous step, minimizing MSE)
5. Run_Lockbox_CNN.py (use best fitted CNN from step 4 to evaluate performance on test set)

In addition, the Results folder contains the saved, fitted 3D-CNN parameters for each fold.

References
Abrol, A., Fu, Z., Salman, M., Silva, R., Du, Y., Plis, S., & Calhoun, V. (2021). Deep learning encodes robust discriminative neuroimaging representations to outperform standard machine learning. Nature communications, 12(1), 353.

The Tabnet model code and results can be found in the DL/Tabnet folder while the machine learning (random forest, elastic net) model code and results are found in the ML_BMI/RF and ML_BMI/EN folders.

