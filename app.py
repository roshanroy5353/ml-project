import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

@st.cache_data
def load_data():
    df = pd.read_csv(
        "road accident ml projec.csv",
        header=None,   
        names=[
            "weather",
            "time_of_day",
            "traffic_density",
            "road_type",
            "road_condition",
            "accident_history",
            "road_safe_to_travel"
        ]  
    )
    return df

df = load_data()


weather_enc = LabelEncoder().fit(df["weather"])
timeofday_enc = LabelEncoder().fit(df["time_of_day"])
traffic_enc = LabelEncoder().fit(df["traffic_density"])
roadtype_enc = LabelEncoder().fit(df["road_type"])
roadcond_enc = LabelEncoder().fit(df["road_condition"])
history_enc = LabelEncoder().fit(df["accident_history"])
safe_enc = LabelEncoder().fit(df["road_safe_to_travel"])


st.title("🚧 Road is Safe or Unsafe")
st.write("Predict whether a road is **safe to travel**.")

st.subheader("Enter Road Details")

weather = st.selectbox("Weather", weather_enc.classes_)
time_of_day = st.selectbox("Time of Day", timeofday_enc.classes_)
traffic = st.selectbox("Traffic Density", traffic_enc.classes_)
road_type = st.selectbox("Road Type", roadtype_enc.classes_)
road_condition = st.selectbox("Road Condition", roadcond_enc.classes_)
accident_history = st.selectbox("Accident History", history_enc.classes_)


input_data = [
weather_enc.transform([weather])[0],
timeofday_enc.transform([time_of_day])[0],
traffic_enc.transform([traffic])[0],
roadtype_enc.transform([road_type])[0],
roadcond_enc.transform([road_condition])[0],
history_enc.transform([accident_history])[0],]


if st.button("Predict Safety"):
    prediction = model.predict([input_data])[0]
    result = safe_enc.inverse_transform([prediction])[0]

    if result.lower() == "yes":
        st.success("✅ The road is SAFE to travel!")
    else:
        st.error("⚠️ The road is NOT SAFE to travel.")


