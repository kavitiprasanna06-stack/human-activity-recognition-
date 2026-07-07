import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load Combined Model
# -------------------------------
with open("har_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
label_encoder = model_data["label_encoder"]

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

st.title("🏃 Human Activity Recognition System")
st.write("Predict human activity using Machine Learning.")

st.markdown("---")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.dataframe(df.head())

        # Remove non-feature columns if present
        if "Activity" in df.columns:
            df = df.drop(columns=["Activity"])

        if "subject" in df.columns:
            df = df.drop(columns=["subject"])

        # Predict
        prediction = model.predict(df)

        predicted_activity = label_encoder.inverse_transform(prediction)

        result = pd.DataFrame({
            "Predicted Activity": predicted_activity
        })

        st.subheader("Prediction Results")
        st.dataframe(result)

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV file.")
