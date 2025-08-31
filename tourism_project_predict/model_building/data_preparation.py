# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/rishabhsinghjk/Tourism-package-predict-dataspace/tourism.csv"
dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")



# Define the target feature for classification
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',                      # Customer's age
    'DurationOfPitch',          # Number of years the customer has been with the bank
    'NumberOfPersonVisiting',   # Customer’s account balance
    'NumberOfFollowups',        # Number of products the customer has with the bank
    'NumberOfTrips',            # Whether the customer has a credit card (binary: 0 or 1)
    'NumberOfChildrenVisiting', # Whether the customer is an active member (binary: 0 or 1)
    'MonthlyIncome'             # Customer’s estimated salary
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # Country where the customer resides
    'CityTier',
    'Occupation',
    'Gender',
    'ProductPitched',
    'PreferredPropertyStar',
    'MaritalStatus',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'Designation'
]

# Define predictor matrix (X) using selected numeric and categorical features
X = dataset[numeric_features + categorical_features]

# Define target variable
y = dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.25,     # 25% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="rishabhsinghjk/Tourism-package-predict-dataspace",
        repo_type="dataset",
    )
