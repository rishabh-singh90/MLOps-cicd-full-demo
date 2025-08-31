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

#Check for duplicate rows, remove them if present
duplicate_count = dataset.duplicated().sum()
if duplicate_count > 0:
    dataset.drop_duplicates(inplace=True)
    print(f"{duplicate_count} duplicate rows removed.")
else:
    print("No duplicate rows found.")

#Dropping CustomerID and Unnamed: 0 columns as they are row identifiers and are not needed in this use-case
dataset.drop(["CustomerID"], axis=1, inplace=True)
dataset.drop(["Unnamed: 0"], axis=1, inplace=True)


# Define the target feature for classification
target = 'ProdTaken'

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

#Creating the train and test csvs
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

#Uploading the train and test files to dataset space
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="rishabhsinghjk/Tourism-package-predict-dataspace",
        repo_type="dataset",
    )
