import pandas as pd
import joblib
import streamlit as st


# Load model and dataset
MODEL_PATH = "models/student_support_model.joblib"
DATA_PATH = "data/student_support_demo.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

TARGET = "support_needed_next_period"


# Page settings
st.set_page_config(
    page_title="Student Support AI",
    page_icon="🎓"
)

st.title("🎓 Student Support AI")

st.write(
    "Enter student information to predict whether "
    "additional academic support may be needed."
)

st.divider()


# Get input columns
input_columns = [
    col for col in df.columns
    if col != TARGET
]


# Remove ID columns
id_columns = [
    col for col in input_columns
    if col.lower() in ["id", "student_id", "studentid"]
]

input_columns = [
    col for col in input_columns
    if col not in id_columns
]


# Student information form
st.subheader("Enter Student Information")

user_input = {}

with st.form("student_form"):

    for column in input_columns:

        if (
            df[column].dtype == "object"
            or str(df[column].dtype) == "category"
            or df[column].dtype == "bool"
        ):

            options = df[column].dropna().unique().tolist()

            user_input[column] = st.selectbox(
                column.replace("_", " ").title(),
                options
            )

        else:

            min_value = float(df[column].min())
            max_value = float(df[column].max())
            default_value = float(df[column].median())

            user_input[column] = st.number_input(
                column.replace("_", " ").title(),
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                step=0.01
            )

    submitted = st.form_submit_button(
        "🔍 Predict Support"
    )


# Prediction
if submitted:

    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]

    st.divider()
    st.subheader("Prediction")

    if prediction == 1:

        st.warning(
            "⚠️ This student may need additional support."
        )

    else:

        st.success(
            "✅ This student may not need additional support."
        )

    # Probability
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_df)[0]

        support_probability = probabilities[1] * 100

        st.write(
            f"Estimated support probability: "
            f"**{support_probability:.1f}%**"
        )

    st.caption(
        "This is an AI-based prediction and should support "
        "educator judgment rather than replace it."
    )