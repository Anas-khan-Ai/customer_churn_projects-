import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu

# PAGE CONFIG
st.set_page_config(
    page_title="AI Churn Prediction System",
    layout="wide",
    page_icon="📊"
)

# DARK THEME CSS
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# LOGIN SYSTEM
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Login System")

    username = st.text_input("Username")
    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()

        else:
            st.error("Invalid Username or Password")

    st.stop()

# LOAD DATA
df = pd.read_csv("data/churn.csv")

# CLEAN DATA
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# SAVE ORIGINAL
df["Churn_Label"] = df["Churn"]

# MAP TARGET
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# LOAD MODEL
model = joblib.load(
    "models/churn_model.pkl"
)

# ENCODE DATA
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# SIDEBAR MENU
with st.sidebar:

    selected = option_menu(
        "Main Menu",
        [
            "Dashboard",
            "Prediction",
            "Insights"
        ],
        icons=[
            "bar-chart",
            "robot",
            "lightbulb"
        ],
        default_index=0
    )

# DASHBOARD
if selected == "Dashboard":

    st.title("📊 AI Customer Churn Dashboard")

    # KPIs
    total_customers = len(df)

    churn_customers = df["Churn"].sum()

    retained_customers = (
        total_customers - churn_customers
    )

    churn_rate = (
        churn_customers / total_customers
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        total_customers
    )

    col2.metric(
        "Churn Customers",
        churn_customers
    )

    col3.metric(
        "Retained Customers",
        retained_customers
    )

    col4.metric(
        "Churn Rate %",
        round(churn_rate, 2)
    )

    # CHURN DISTRIBUTION
    fig1 = px.histogram(
        df,
        x="Churn_Label",
        title="Customer Churn Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # MONTHLY CHARGES
    fig2 = px.box(
        df,
        x="Churn_Label",
        y="MonthlyCharges",
        title="Monthly Charges vs Churn"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # TENURE ANALYSIS
    fig3 = px.histogram(
        df,
        x="tenure",
        color="Churn_Label",
        title="Customer Tenure Analysis"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # DOWNLOAD REPORT
    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name="churn_report.csv",
        mime="text/csv"
    )

# PREDICTION PAGE
elif selected == "Prediction":

    st.title("🤖 Customer Churn Prediction")

    tenure = st.slider(
        "Tenure",
        1,
        72,
        12
    )

    monthly_charges = st.slider(
        "Monthly Charges",
        10,
        150,
        70
    )

    total_charges = st.slider(
        "Total Charges",
        10,
        10000,
        1000
    )

    contract = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    if st.button("Predict Churn"):

        contract_map = {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }

        internet_map = {
            "DSL": 0,
            "Fiber optic": 1,
            "No": 2
        }

        input_data = [[
            0,
            0,
            0,
            0,
            tenure,
            1,
            1,
            1,
            internet_map[internet_service],
            1,
            1,
            1,
            1,
            1,
            1,
            contract_map[contract],
            1,
            1,
            "0",
            monthly_charges,
            total_charges
        ]]

        prediction = model.predict(input_data)

        if prediction[0] == 1:

            st.error(
                "⚠ High Risk Customer Likely To Churn"
            )

        else:

            st.success(
                "✅ Customer Likely To Stay"
            )

# INSIGHTS PAGE
elif selected == "Insights":

    st.title("💡 Business Insights")

    st.info("""
    Key Insights:
    
    • Customers with month-to-month contracts are more likely to churn.
    
    • High monthly charges increase churn probability.
    
    • Long tenure customers are more loyal.
    
    • Fiber optic users showed higher churn rates.
    
    • Retention offers can reduce churn.
    """)

    st.subheader("Recommended Retention Strategies")

    st.write("""
    ✅ Loyalty rewards
    
    ✅ Long-term contract discounts
    
    ✅ Personalized offers
    
    ✅ Better customer support
    
    ✅ Subscription benefits
    """)