The goal of this competition is to predict patient survival using their medical history.

NN testing.py - testing out neural networks for this task (AUC ~ 0.84)

Test_sandbox.py - for testing different sklearn model and hyperparameters

Transforemer.pkl - pickled version of Preprocessor class's instance. It's used for raw data preprocessing.

model.py - final model, which the combination of of different models

preprocessor.py - for raw input data preprocessing

run_pipeline.py - for runing the final model. Input gores through prepricessor(transformer) and then the model makes its prediction.


# Hospital_Adv-ML

dependencies`

numpy==1.24.2

pandas==2.0.0

scikit-learn==1.2.2

python==3.10

xgboost==1.7.5
