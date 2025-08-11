import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved model and preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Load dataset to get dropdown values for Area and Item
df = pd.read_csv("yield_df.CSV")
df.drop('Unnamed: 0', axis=1, inplace=True)

# Clean dataset like in backend
def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True

to_drop = df[df["average_rain_fall_mm_per_year"].apply(isStr)].index
df = df.drop(to_drop)

areas = sorted(df['Area'].unique())
items = sorted(df['Item'].unique())

# Streamlit UI
st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

st.title("🌾 Crop Yield Prediction App")
st.markdown("Predict crop yield (`hg/ha`) based on environmental and agricultural factors.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2020, step=1)
    avg_rain = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=50.0, step=0.1)
    pesticides = st.number_input("Pesticides Used (tonnes)", min_value=0.0, value=1000.0, step=0.1)

with col2:
    avg_temp = st.number_input("Average Temperature (°C)", min_value=-20.0, max_value=60.0, value=25.0, step=0.1)
    area = st.selectbox("Country / Area", areas)
    item = st.selectbox("Crop Item", items)

# Prediction button
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: green;
        color: white;
    }
    div.stButton > button:first-child:hover {
        background-color: darkgreen;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

if st.button("Predict Yield", use_container_width=True):
        features = np.array([[year, avg_rain, pesticides, avg_temp, area, item]])
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features)[0]

        st.metric(label="Predicted Crop Yield (hg/ha)", value=f"{prediction:,.2f}")
st.markdown("---")
st.caption("Built by Ali Raza")
