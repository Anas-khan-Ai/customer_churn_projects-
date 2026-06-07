import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA
df = pd.read_csv("data/churn.csv")

# DATA CLEANING
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# TARGET COLUMN
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print(df.head())

print(df.info())

print(df.describe())

# CHURN COUNT
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Count")
plt.show()

# MONTHLY CHARGES
sns.boxplot(
    x="Churn",
    y="MonthlyCharges",
    data=df
)
plt.title("Monthly Charges vs Churn")
plt.show()

# TENURE ANALYSIS
sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    kde=True
)
plt.title("Tenure Distribution")
plt.show()