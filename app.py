import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the trained models
fault_model = pickle.load(open('fault_model.pkl','rb'))
rework_model = pickle.load(open('rework_model.pkl','rb'))

def predict_vehicle_loss(pol_num, vehicle_type, fault_name, fault_time, defect_code, rework_time):
    # Create LabelEncoder objects
    le_pol_num = LabelEncoder()
    le_vehicle_type = LabelEncoder()

    # Encode categorical features
    pol_num_encoded = le_pol_num.fit_transform([pol_num])
    vehicle_type_encoded = le_vehicle_type.fit_transform([vehicle_type])

    # Create input arrays for both models
    fault_input = np.array([[pol_num_encoded[0], vehicle_type_encoded[0], fault_name, fault_time]])
    rework_input = np.array([[pol_num_encoded[0], vehicle_type_encoded[0], defect_code, rework_time]])

    # Make predictions for both models
    fault_prediction = fault_model.predict(fault_input)
    rework_prediction = rework_model.predict(rework_input)

    # Format predictions
    predicted_fault_loss = int(fault_prediction[0])
    predicted_rework_loss = int(rework_prediction[0])

    return predicted_fault_loss, predicted_rework_loss

def main():
    st.title("Vehicle Loss Prediction App")

    # Get user input
    pol_num = st.text_input("Enter the Polishing Line:")
    vehicle_type = st.text_input("Enter the Vehicle Type:")
    fault_name = st.number_input("Enter the Fault Name:", min_value=0, step=1)
    fault_time = st.number_input("Enter the Fault Time (minutes):", min_value=0, step=1)
    defect_code = st.number_input("Enter the Defect Code:", min_value=0, step=1)
    rework_time = st.number_input("Enter the Rework Time (minutes):", min_value=0, step=1)

    # Predict vehicle loss
    if st.button("Predict"):
        fault_loss, rework_loss = predict_vehicle_loss(pol_num, vehicle_type, fault_name, fault_time, defect_code, rework_time)
        st.write(f"Predicted Fault Vehicle Loss: {fault_loss}")
        st.write(f"Predicted Rework Vehicle Loss: {rework_loss}")
        st.write(f"Total Predicted Vehicle Loss: {fault_loss + rework_loss}")

if __name__ == '__main__':
    main()