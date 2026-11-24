import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.sparse import hstack
from textwrap import dedent


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# 2. HTML RENDERING
# ==================================================

def render_html(content):
    st.html(dedent(content))


# ==================================================
# 3. CUSTOM CSS
# ==================================================

render_html("""
<style>

.stApp {
    background: linear-gradient(135deg, #f5f3ff, #eef2ff);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #312e81;
}

.dashboard-header {
    background: linear-gradient(
        120deg,
        #4f46e5,
        #7c3aed,
        #9333ea
    );
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 28px;
    box-shadow: 0 5px 20px rgba(79,70,229,0.2);
}

.dashboard-header h1 {
    color: white;
    font-size: 34px;
}

.dashboard-header p {
    color: #ede9fe;
    font-size: 16px;
}

.metric-card {
    background: white;
    padding: 23px;
    border-radius: 15px;
    border-left: 5px solid #7c3aed;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
    min-height: 125px;
}

.metric-title {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #312e81;
    font-size: 28px;
    font-weight: bold;
    margin-top: 12px;
}

.info-card {
    background: white;
    padding: 23px;
    border-radius: 15px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
    margin-bottom: 18px;
    min-height: 150px;
}

.info-card h3 {
    color: #7c3aed;
    font-size: 21px;
}

.info-card p {
    color: #475569;
    line-height: 1.7;
}

.section-heading {
    color: #312e81;
    font-size: 24px;
    font-weight: bold;
    margin-top: 30px;
    margin-bottom: 20px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #6d28d9, #4f46e5);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 11px 22px;
    font-weight: bold;
}

[data-testid="stSidebar"] {
    background: #ede9fe;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: #4c1d95;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    margin-top: 40px;
    border-top: 1px solid #ddd6fe;
}

</style>
""")


# ==================================================
# 4. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RISK_FILE = (
    BASE_DIR / "reports" / "customer_churn_risk_scores.csv"
)

RESULTS_FILE = (
    BASE_DIR / "reports" / "prediction_results.csv"
)


# ==================================================
# 5. LOAD MODEL AND PREPROCESSING OBJECTS
# ==================================================

@st.cache_resource
def load_model():

    model_path = (
        BASE_DIR / "models" / "tuned_logistic_regression.pkl"
    )

    return joblib.load(model_path)


@st.cache_resource
def load_preprocessing():

    preprocessing_path = (
        BASE_DIR / "models" / "preprocessing_objects.joblib"
    )

    return joblib.load(preprocessing_path)


try:

    model = load_model()

    preprocessing_objects = load_preprocessing()

    scaler = preprocessing_objects["scaler"]

    encoder = preprocessing_objects["encoder"]

    numerical_features = preprocessing_objects[
        "numerical_features"
    ]

    categorical_features = preprocessing_objects[
        "categorical_features"
    ]

except Exception as e:

    st.error(f"Unable to load model files: {e}")
    st.stop()


# ==================================================
# 6. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("📊 Telecom Analytics")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Predict Churn",
        "Customer Insights",
        "Business Summary",
        "About the Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Machine Learning Model Loaded")

st.sidebar.markdown("### Project Information")

st.sidebar.write("**Model:** Logistic Regression")

st.sidebar.write("**Dataset:** IBM Telco Customer Churn")

st.sidebar.write("**Task:** Binary Classification")


# ==================================================
# 7. REUSABLE HEADER
# ==================================================

def show_header():

    render_html("""
    <div class="dashboard-header">

        <h1>📊 Telecom Customer Churn Prediction</h1>

        <p>
        Understand customer behaviour, identify churn risks,
        and support better customer retention decisions.
        </p>

    </div>
    """)


# ==================================================
# 8. METRIC CARDS
# ==================================================

def show_metrics():

    metrics = [
        ("Total Customers", "7,043"),
        ("Overall Churn Rate", "26.54%"),
        ("Retained Customers", "5,174"),
        ("Churned Customers", "1,869")
    ]

    columns = st.columns(4)

    for column, metric in zip(columns, metrics):

        with column:

            render_html(f"""
            <div class="metric-card">

                <div class="metric-title">
                    {metric[0]}
                </div>

                <div class="metric-value">
                    {metric[1]}
                </div>

            </div>
            """)


# ==================================================
# 9. REUSABLE INFORMATION CARD
# ==================================================

def show_info_card(title, value, description):

    render_html(f"""
    <div class="info-card">

        <h3>{title}</h3>

        <h2>{value}</h2>

        <p>{description}</p>

    </div>
    """)


# ==================================================
# 10. DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    show_header()

    show_metrics()

    render_html("""
    <div class="section-heading">
        Customer Churn Overview
    </div>
    """)

    col1, col2 = st.columns(2)

    # Customer distribution chart

    with col1:

        st.markdown("### Customer Distribution")

        churn_data = pd.DataFrame({
            "Customer Type": ["Churned", "Retained"],
            "Customers": [1869, 5174]
        })

        st.bar_chart(
            churn_data,
            x="Customer Type",
            y="Customers"
        )

    # Churn percentage chart

    with col2:

        st.markdown("### Churn Percentage")

        percentage_data = pd.DataFrame({
            "Customer Type": ["Churned", "Retained"],
            "Percentage": [26.54, 73.46]
        })

        st.bar_chart(
            percentage_data,
            x="Customer Type",
            y="Percentage"
        )

    # Donut chart

    st.markdown("### 🍩 Customer Retention Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.pie(
        [5174, 1869],
        labels=["Retained", "Churned"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.4},
        pctdistance=0.8
    )

    ax.set_title("Customer Retention vs Churn")

    st.pyplot(fig)

    plt.close(fig)

    # Customer risk distribution

    render_html("""
    <div class="section-heading">
        🎯 Customer Risk Distribution
    </div>
    """)

    if RISK_FILE.exists():

        risk_df = pd.read_csv(RISK_FILE)

        if "Churn_Probability" in risk_df.columns:

            risk_df["Risk_Category"] = risk_df[
                "Churn_Probability"
            ].apply(
                lambda probability:
                    "High Risk" if probability >= 0.70
                    else "Medium Risk" if probability >= 0.30
                    else "Low Risk"
            )

            risk_counts = (
                risk_df["Risk_Category"]
                .value_counts()
                .reindex(
                    ["High Risk", "Medium Risk", "Low Risk"],
                    fill_value=0
                )
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "🔴 High Risk",
                int(risk_counts["High Risk"])
            )

            col2.metric(
                "🟠 Medium Risk",
                int(risk_counts["Medium Risk"])
            )

            col3.metric(
                "🟢 Low Risk",
                int(risk_counts["Low Risk"])
            )

            risk_chart_data = pd.DataFrame({
                "Risk Category": risk_counts.index,
                "Customers": risk_counts.values
            })

            st.bar_chart(
                risk_chart_data,
                x="Risk Category",
                y="Customers"
            )

        else:

            st.error(
                "Churn_Probability column is missing from the risk CSV."
            )

    else:

        st.warning("Customer risk score file not found.")

    # Business observations

    render_html("""
    <div class="section-heading">
        Key Business Observations
    </div>
    """)

    col1, col2, col3 = st.columns(3)

    with col1:

        show_info_card(
            "📅 Contract Type",
            "42.71%",
            "Churn rate among month-to-month contract customers."
        )

    with col2:

        show_info_card(
            "💳 Payment Method",
            "45.29%",
            "Churn rate among customers using electronic checks."
        )

    with col3:

        show_info_card(
            "🌐 Internet Service",
            "41.89%",
            "Churn rate among fiber optic internet customers."
        )

    render_html("""
    <div class="section-heading">
        Machine Learning Model Information
    </div>
    """)

    col1, col2, col3 = st.columns(3)

    col1.info("Algorithm: Logistic Regression")

    col2.info("Problem Type: Classification")

    col3.info("Input Features: 19")


# ==================================================
# 11. CUSTOMER CHURN PREDICTION PAGE
# ==================================================

elif page == "Predict Churn":

    show_header()

    st.markdown("## 🔍 Predict Customer Churn")

    st.write(
        "Enter customer details to estimate their likelihood "
        "of leaving the telecom service."
    )

    st.markdown("---")

    # Customer input form

    with st.form("customer_prediction_form"):

        st.markdown("### 👤 Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

        with col2:

            senior_citizen_option = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"]
            )

            senior_citizen = (
                1 if senior_citizen_option == "Yes" else 0
            )

        with col3:

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

        with col2:

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=12
            )

        with col3:

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

        st.markdown("### 📡 Service Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

        with col2:

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

        with col3:

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

        with col2:

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col3:

            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=2000.0,
                value=70.0
            )

        st.markdown("### 🛠️ Additional Services")

        col1, col2, col3 = st.columns(3)

        with col1:

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col2:

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

        with col3:

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=800.0
        )

        st.markdown("---")

        submitted = st.form_submit_button(
            "🔮 Predict Customer Churn",
            use_container_width=True
        )

    # --------------------------------------------------
    # PREDICTION LOGIC
    # --------------------------------------------------

    if submitted:

        customer_data = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [senior_citizen],

            "Partner": [partner],

            "Dependents": [dependents],

            "tenure": [tenure],

            "PhoneService": [phone_service],

            "MultipleLines": [multiple_lines],

            "InternetService": [internet_service],

            "OnlineSecurity": [online_security],

            "OnlineBackup": [online_backup],

            "DeviceProtection": [device_protection],

            "TechSupport": [tech_support],

            "StreamingTV": [streaming_tv],

            "StreamingMovies": [streaming_movies],

            "Contract": [contract],

            "PaperlessBilling": [paperless_billing],

            "PaymentMethod": [payment_method],

            "MonthlyCharges": [monthly_charges],

            "TotalCharges": [total_charges]

        })

        try:

            # Arrange columns in training order

            customer_data = customer_data[
                numerical_features + categorical_features
            ]

            # Scale numerical features

            customer_numerical = scaler.transform(
                customer_data[numerical_features]
            )

            # Encode categorical features

            customer_categorical = encoder.transform(
                customer_data[categorical_features]
            )

            # Combine processed features

            customer_processed = hstack([
                customer_numerical,
                customer_categorical
            ]).tocsr()

            # Generate prediction

            prediction = model.predict(
                customer_processed
            )[0]

            probabilities = model.predict_proba(
                customer_processed
            )[0]

            yes_index = list(model.classes_).index("Yes")

            churn_probability = probabilities[yes_index]

            # Determine risk category

            if churn_probability >= 0.70:

                risk_level = "High Risk"

            elif churn_probability >= 0.30:

                risk_level = "Medium Risk"

            else:

                risk_level = "Low Risk"

            # Save customer prediction results

            prediction_record = pd.DataFrame([{
                "Gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "Tenure": tenure,
                "Contract": contract,
                "InternetService": internet_service,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "Churn_Probability": round(churn_probability, 4),
                "Prediction": prediction,
                "Risk_Category": risk_level
            }])

            prediction_record.to_csv(
                RESULTS_FILE,
                mode="a",
                header=(
                    not RESULTS_FILE.exists()
                    or RESULTS_FILE.stat().st_size == 0
                ),
                index=False
            )

            # Set risk styling and recommendation

            if risk_level == "High Risk":

                risk_color = "#fee2e2"
                text_color = "#991b1b"
                risk_icon = "🔴"

                recommendation = (
                    "Prioritize this customer for a personalized "
                    "retention campaign. Review their service "
                    "experience and offer suitable support."
                )

            elif risk_level == "Medium Risk":

                risk_color = "#fef3c7"
                text_color = "#92400e"
                risk_icon = "🟠"

                recommendation = (
                    "Monitor customer satisfaction and engagement. "
                    "Consider proactive communication and suitable offers."
                )

            else:

                risk_color = "#dcfce7"
                text_color = "#166534"
                risk_icon = "🟢"

                recommendation = (
                    "Continue regular customer engagement and "
                    "maintain a positive service experience."
                )

            # Prediction outcome

            if prediction == "Yes":

                outcome = "Likely to Churn"
                outcome_icon = "⚠️"

            else:

                outcome = "Likely to Stay"
                outcome_icon = "✅"

            # Display prediction report

            st.markdown("---")

            st.markdown("## 📊 Customer Prediction Report")

            render_html(f"""
            <div style="
                background: white;
                padding: 30px;
                border-radius: 18px;
                border-left: 7px solid {text_color};
                box-shadow: 0 4px 18px rgba(0,0,0,0.07);
                margin-bottom: 20px;
            ">

                <h2 style="color: #312e81;">
                    {outcome_icon} {outcome}
                </h2>

                <p style="color: #64748b;">
                    Machine Learning Prediction Result
                </p>

                <h1 style="
                    color: {text_color};
                    font-size: 42px;
                ">
                    {churn_probability * 100:.2f}%
                </h1>

                <p style="color: #64748b;">
                    Estimated Churn Probability
                </p>

                <div style="
                    background: {risk_color};
                    color: {text_color};
                    padding: 12px 18px;
                    border-radius: 10px;
                    display: inline-block;
                    font-weight: bold;
                ">
                    {risk_icon} {risk_level}
                </div>

            </div>
            """)

            st.progress(float(churn_probability))

            # Business recommendation

            st.markdown("### 💡 Recommended Business Action")

            render_html(f"""
            <div class="info-card">

                <h3>Retention Recommendation</h3>

                <p>{recommendation}</p>

            </div>
            """)

            st.success(
                "Prediction successfully saved to prediction_results.csv"
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")


# ==================================================
# 12. CUSTOMER INSIGHTS PAGE
# ==================================================

elif page == "Customer Insights":

    show_header()

    st.markdown("## 📈 Customer Behaviour Insights")

    insights = [

        (
            "📅 Contract Type",
            "42.71%",
            "Churn rate among month-to-month contract customers."
        ),

        (
            "💳 Payment Method",
            "45.29%",
            "Churn rate among electronic check customers."
        ),

        (
            "🌐 Internet Service",
            "41.89%",
            "Churn rate among fiber optic customers."
        ),

        (
            "🔐 Online Security",
            "41.77%",
            "Churn rate among customers without online security."
        ),

        (
            "🛠️ Technical Support",
            "41.64%",
            "Churn rate among customers without technical support."
        ),

        (
            "⏳ Customer Tenure",
            "17.98 months",
            "Average tenure of churned customers, compared with "
            "37.57 months for retained customers."
        )

    ]

    for i in range(0, len(insights), 3):

        columns = st.columns(3)

        for column, item in zip(
            columns,
            insights[i:i + 3]
        ):

            with column:

                show_info_card(
                    item[0],
                    item[1],
                    item[2]
                )

    st.caption(
        "These findings represent associations in the dataset "
        "and do not establish causation."
    )


# ==================================================
# 13. BUSINESS SUMMARY PAGE
# ==================================================

elif page == "Business Summary":

    show_header()

    st.markdown("## 💡 Business Recommendations")

    recommendations = [

        (
            "📅 Improve Contract Retention",
            "Encourage month-to-month customers to explore "
            "longer-term contracts through suitable offers."
        ),

        (
            "💳 Review Payment Experience",
            "Investigate issues associated with electronic checks "
            "and encourage convenient payment alternatives."
        ),

        (
            "🌐 Improve Fiber Optic Experience",
            "Review service quality, pricing and customer satisfaction "
            "among fiber optic users."
        ),

        (
            "🔐 Promote Additional Services",
            "Introduce relevant online security and technical "
            "support packages."
        ),

        (
            "⏳ Focus on New Customers",
            "Provide onboarding support and early engagement "
            "during the first months of a customer relationship."
        ),

        (
            "🤖 Use Predictive Risk Scores",
            "Prioritize high-risk customers for personalized "
            "retention campaigns."
        )

    ]

    for i in range(0, len(recommendations), 2):

        columns = st.columns(2)

        for column, item in zip(
            columns,
            recommendations[i:i + 2]
        ):

            with column:

                render_html(f"""
                <div class="info-card">

                    <h3>{item[0]}</h3>

                    <p>{item[1]}</p>

                </div>
                """)

    st.success(
        "Business Goal: Identify at-risk customers early "
        "and support customer retention."
    )


# ==================================================
# 14. ABOUT THE PROJECT PAGE
# ==================================================

elif page == "About the Project":

    show_header()

    st.markdown("## ℹ️ About This Project")

    st.write("""
    Telecom Customer Churn Prediction is a machine learning
    project designed to identify customers who may discontinue
    their telecom services.

    The project uses the IBM Telco Customer Churn dataset and
    applies data analysis, preprocessing, model training,
    evaluation and customer risk scoring.
    """)

    st.markdown("### 🎯 Project Objectives")

    st.markdown("""
    - Understand customer behaviour.
    - Identify factors associated with customer churn.
    - Build a machine learning classification model.
    - Predict the probability of customer churn.
    - Categorize customers based on churn risk.
    - Provide business recommendations for retention.
    """)

    st.markdown("### 🧠 Technologies Used")

    col1, col2, col3 = st.columns(3)

    col1.info("🐍 Python")

    col2.info("🐼 Pandas and NumPy")

    col3.info("🤖 Scikit-learn")

    col1, col2, col3 = st.columns(3)

    col1.info("📊 Matplotlib")

    col2.info("📈 Seaborn")

    col3.info("🌐 Streamlit")

    st.markdown("### 🤖 Machine Learning Model")

    st.write("**Selected Model:** Logistic Regression")

    st.write("**Problem Type:** Binary Classification")

    st.write("**Target Variable:** Churn")

    st.write("**Classes:** Yes / No")

    st.markdown("### 📂 Project Structure")

    st.code("""
Telecom-Customer-Churn-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_building.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_model_improvement.ipynb
│   └── 06_business_insights_and_recommendations.ipynb
│
├── models/
│   ├── tuned_logistic_regression.pkl
│   └── preprocessing_objects.joblib
│
├── reports/
│
└── app/
    └── app.py
    """)


# ==================================================
# 15. FOOTER
# ==================================================

render_html("""
<div class="footer">

    <h3>📊 Telecom Customer Churn Prediction</h3>

    <p>
    Machine Learning | Customer Analytics | Business Intelligence
    </p>

    <p>
    Built using Python and Streamlit
    </p>

</div>
""")