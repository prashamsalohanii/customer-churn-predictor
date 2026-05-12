import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict if they will churn.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account Info")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", [0, 1, 2], format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
    paperless = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    payment = st.selectbox("Payment Method", [0, 1, 2, 3], format_func=lambda x: ["Bank transfer", "Credit card", "Electronic check", "Mailed check"][x])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: ["Female", "Male"][x])

with col2:
    st.subheader("Services")
    phone = st.selectbox("Phone Service", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    multiple_lines = st.selectbox("Multiple Lines", [0, 1, 2], format_func=lambda x: ["No", "Yes", "No phone"][x])
    internet = st.selectbox("Internet Service", [0, 1, 2], format_func=lambda x: ["DSL", "Fiber optic", "No"][x])
    online_sec = st.selectbox("Online Security", [0, 1, 2], format_func=lambda x: ["No", "Yes", "No internet"][x])
    online_backup = st.selectbox("Online Backup", [0, 1, 2], format_func=lambda x: ["No", "Yes", "No internet"][x])
    device_protection = st.selectbox("Device Protection", [0, 1, 2], format_func=lambda x: ["No", "Yes", "No internet"][x])

with col3:
    st.subheader("More Services")
    tech_support = st.selectbox("Tech Support", [0, 1, 2], format_func=lambda x: ["No", "Yes", "No internet"][x])
    streaming_tv = st.selectbox("Streaming TV", [0, 1, 2], format_func=lambda x: ["No", "Yes", "No internet"][x])
    streaming_movies = st.selectbox("Streaming Movies", [0, 1, 2], format_func=lambda x: ["No", "Yes", "No internet"][x])
    partner = st.selectbox("Partner", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    dependents = st.selectbox("Dependents", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

st.divider()

if st.button("Predict Churn", use_container_width=True):
    input_dict = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': online_sec,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    input_data = pd.DataFrame([input_dict])[feature_names]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    col_a, col_b = st.columns(2)

    with col_a:
        if prediction == 1:
            st.error("⚠️ This customer is LIKELY TO CHURN!")
            st.markdown("**Suggestions to retain this customer:**")
            st.markdown("- Offer a discount or loyalty reward")
            st.markdown("- Upgrade their internet plan")
            st.markdown("- Assign a dedicated support agent")
        else:
            st.success("✅ This customer is NOT likely to churn.")
            st.markdown("**This customer is happy! Keep up the good service.**")

    with col_b:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability[1] * 100, 1),
            title={'text': "Churn Probability %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if prediction == 1 else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "#d4f5d4"},
                    {'range': [40, 70], 'color': "#fff3cd"},
                    {'range': [70, 100], 'color': "#ffd5d5"}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Top Factors Affecting Churn")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(10)

    fig2 = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#534AB7'
    ))
    fig2.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)
