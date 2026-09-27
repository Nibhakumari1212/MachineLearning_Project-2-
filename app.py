import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

DATA_PATH = "diabetes_prediction_dataset.csv"

# Same encoding jo notebook mein LabelEncoder ne banaya tha (alphabetical order)
GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}
SMOKING_MAP = {"No Info": 0, "current": 1, "ever": 2, "former": 3, "never": 4, "not current": 5}


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)

    df["gender"] = df["gender"].map(GENDER_MAP)
    df["smoking_history"] = df["smoking_history"].map(SMOKING_MAP)

    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return clf, scaler, metrics


clf, scaler, metrics = train_model()

st.title("🩺 Diabetes Prediction App")
st.write("Logistic Regression model — apni health details daal kar diabetes risk check karein.")

with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    st.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    st.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    st.metric("F1 Score", f"{metrics['f1'] * 100:.2f}")
    st.caption("Recall zyada important hai — kisi diabetic patient ko miss nahi karna chahiye.")

st.subheader("Patient Details")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with col2:
    smoking_history = st.selectbox("Smoking History", list(SMOKING_MAP.keys()))
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([{
        "gender": GENDER_MAP[gender],
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": SMOKING_MAP[smoking_history],
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
    }])

    input_scaled = scaler.transform(input_df)
    prediction = clf.predict(input_scaled)[0]
    probability = clf.predict_proba(input_scaled)[0]

    st.write("---")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes — {probability[1] * 100:.1f}% probability")
    else:
        st.success(f"✅ Low Risk of Diabetes — {probability[0] * 100:.1f}% probability")

    st.caption("Note: Ye sirf ek ML model ka prediction hai, medical diagnosis nahi. Doctor se consult zaroor karein.")
