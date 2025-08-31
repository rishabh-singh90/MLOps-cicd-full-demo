# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

#Setting the tracking URL and experiment name in mlflow
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("MLOps_experiment_tour_prod")

api = HfApi()

#loading the train and test sets from hugging face dataset space

Xtrain_path = "hf://datasets/rishabhsinghjk/Tourism-package-predict-dataspace/Xtrain.csv"
Xtest_path = "hf://datasets/rishabhsinghjk/Tourism-package-predict-dataspace/Xtest.csv"
ytrain_path = "hf://datasets/rishabhsinghjk/Tourism-package-predict-dataspace/ytrain.csv"
ytest_path = "hf://datasets/rishabhsinghjk/Tourism-package-predict-dataspace/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# List of numerical features in the dataset
numeric_features = [
    'Age',                      # Customer's age
    'DurationOfPitch',          # The time duration for which the product was explained to the customer
    'NumberOfPersonVisiting',   # Expected number of persons visiting along with customer
    'NumberOfFollowups',        # Number of followups done with the customer after the pitch
    'NumberOfTrips',            # Average number of trips done by customer per year
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer
    'MonthlyIncome'             # Customer’s estimated monthly salary
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',          # Country where the customer resides
    'CityTier',               # City tier 1, 2 or 3
    'Occupation',             # Source of income of customer
    'Gender',                 # Gender of customer
    'ProductPitched',         # The category of the product piched to the customer
    'PreferredPropertyStar',  # The property rating preference of the customer
    'MaritalStatus',          # Marital status of customer
    'Passport',               # Does customer has passport or not
    'PitchSatisfactionScore', # The satisfaction rating of customer for the piched product
    'OwnCar',                 # Does the customer owns a car
    'Designation'             # What is the designation of customer
]


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [ 120, 130, 140, 150],    # number of tree to build
    'xgbclassifier__max_depth': [3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [ 0.6, 0.7],    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [ 0.6, 0.7],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.6, 0.7],    # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.55

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Saving the model locally first
    model_path = "best_tour_pkg_predct_v1.joblib"
    joblib.dump(best_model, model_path)

    # Logging the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # defining repo model to Hugging Face
    repo_id = "rishabhsinghjk/Tourism-package-predict-model"
    repo_type = "model"

    # Step 1: Check if the model space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # Step 2: Upload the model to HF model space just created
    print(f"Uploading model to space '{repo_id}'...")
    api.upload_file(
        path_or_fileobj="best_tour_pkg_predct_v1.joblib",
        path_in_repo="best_tour_pkg_predct_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
