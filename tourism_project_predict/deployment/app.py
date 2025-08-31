import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model space
model_path = hf_hub_download(repo_id="rishabhsinghjk/Tourism-package-predict-model", filename="best_tour_pkg_predct_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism package Customer Acceptance Prediction
st.title("Tourism package Customer Acceptance Prediction App")
st.write("The Tourism package Customer Acceptance Prediction App is an internal tool for tourism comapny employees to predicts whether customers will accept the pitched package based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to accept.")

# Collect user input
DurationOfPitch = st.number_input("Duration Of Pitch (Time duration in minutes)", min_value=1.0, max_value=100.0, value=1.0)
TypeofContact = st.selectbox("Type of Contact (Method by which the customer was contacted)", ["Self Enquiry", "Company Invited"])
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=25)
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting",min_value=1, value=2)
NumberOfFollowups = st.number_input("Number Of Followups (Follow-ups done with the customer)", min_value=0.0, value=1.0)
NumberOfTrips = st.number_input("Number Of Trips (Number of trips the customer takes annually.)", min_value=0.0, value=1.0)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, value=0)
MonthlyIncome = st.number_input("Monthly Income of customer", min_value=100.0, value=10000.0)
CityTier = st.selectbox("City Tier of customer", ["1", "2", "3"])
Occupation = st.selectbox("Occupation of customer", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product category Pitched to customer", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", ["1", "2", "3", "4", "5"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score by  customer", ["5", "4", "3", "2", "1"])
OwnCar = st.selectbox("Customer owns Car?", ["Yes", "No"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])



# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'DurationOfPitch': DurationOfPitch,
    'TypeofContact': TypeofContact,
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'Designation': Designation
}])

# Set the classification threshold
classification_threshold = 0.55

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "accept" if prediction == 1 else "not accept"
    st.write(f"Based on the information provided, the customer is likely to {result} package.")
