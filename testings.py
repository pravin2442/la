# Decision Tree Models: ID3, C4.5, CART (All-in-One)
# ---------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                   'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Step 2: Preprocess (encode)
X = pd.get_dummies(df[['Outlook', 'Temperature', 'Humidity', 'Wind']])
y = df['PlayTennis'].map({'No': 0, 'Yes': 1})

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Choose algorithm by criterion:
# "entropy" → ID3 or C4.5
# "gini" → CART
criterion_choice = "entropy"   # change to "gini" for CART

# Step 5: Train model
model = DecisionTreeClassifier(criterion=criterion_choice, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print(f"Algorithm Used: {'CART (Gini)' if criterion_choice=='gini' else 'ID3/C4.5 (Entropy)'}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Display tree rules
rules = export_text(model, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n", rules)

# Step 8: Plot tree
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=list(X.columns), class_names=['No', 'Yes'],
          filled=True, rounded=True)
plt.title(f"Decision Tree ({criterion_choice.upper()})")
plt.show()












# ................................................................................................................
import pandas as pd

# Step 1: Load your dataset
df = pd.read_csv("your_dataset.csv")   # or pd.read_excel("your_dataset.xlsx")

# Step 2: Check for missing values
print("Missing values in each column:\n")
print(df.isnull().sum())

# Step 3: Drop irrelevant columns (optional)
# Example: dropping ID, Name, or other useless columns
df.drop(['ColumnName1', 'ColumnName2'], axis=1, inplace=True)

# Step 4: Handle missing values
# Fill numeric columns (like Age, Price) using mean or median
df['Age'].fillna(df['Age'].median(), inplace=True)     # or mean()

# Fill categorical columns (like Gender, Embarked) using mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Step 5: Remove duplicates (optional)
df.drop_duplicates(inplace=True)

# Step 6: Remove invalid or negative values (for numerical columns)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Step 7: Confirm cleaning
print("\nAfter cleaning:")
print(df.info())
print("\nAny missing values left?")
print(df.isnull().sum())
# ........................................................................................


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned dataset
df = pd.read_csv("your_dataset.csv")

# 1️⃣ Histogram — Distribution of a numerical column
plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 2️⃣ Count Plot — Count of categories
sns.countplot(data=df, x='Sex')
plt.title('Count of Passengers by Gender')
plt.show()

# 3️⃣ Bar Plot — Comparison of averages
sns.barplot(data=df, x='Pclass', y='Survived')
plt.title('Survival Rate by Passenger Class')
plt.show()

# 4️⃣ Box Plot — Spread of values across groups
sns.boxplot(data=df, x='Survived', y='Age')
plt.title('Age vs Survival')
plt.show()

# 5️⃣ Heatmap — Correlation between numerical features
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 6️⃣ Pie Chart — Percentage distribution
df['Embarked'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Passengers by Embarked Port')
plt.ylabel('')
plt.show()
# ........................................................................................


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("Online_Retail.xlsx")

# Histogram for Quantity
sns.histplot(df['Quantity'], bins=30, kde=True)
plt.title('Quantity Distribution')
plt.show()

# Boxplot for Unit Price
sns.boxplot(x=df['UnitPrice'])
plt.title('Unit Price Boxplot')
plt.show()

# Bar chart — Top 10 customers by total spending
df['TotalSales'] = df['Quantity'] * df['UnitPrice']
top_customers = df.groupby('CustomerID')['TotalSales'].sum().sort_values(ascending=False).head(10)
top_customers.plot(kind='bar', color='green')
plt.title('Top 10 Customers by Spending')
plt.xlabel('Customer ID')
plt.ylabel('Total Sales')
plt.show()

# Line plot — Monthly Sales Trend
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.to_period('M')
monthly_sales = df.groupby('Month')['TotalSales'].sum()
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

# ................................................................................................

# 10 random integers between 1 and 100
random_ints = np.random.randint(1, 100, 10)
print("Random Integers:", random_ints)

# 5 random decimal (float) numbers between 0 and 1
random_floats = np.random.rand(5)
print("Random Floats:", random_floats)
# 3 rows × 4 columns matrix of random numbers between 0 and 10
data = np.random.randint(0, 10, (3, 4))
print(data)

# ....................................................................................................
# first program

# Titanic Descriptive Analytics - Simple Version
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Import and Explore Dataset
df = pd.read_csv("Titanic.csv")

print("First 5 Rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe())

# 2. Clean the Dataset
# Drop irrelevant columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)      # Median imputation
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Mode imputation

# 3. Descriptive Statistics
print("\nAge Mean:", df['Age'].mean())
print("Age Median:", df['Age'].median())
print("Age Mode:", df['Age'].mode()[0])

print("\nValue Counts:")
print("Sex:\n", df['Sex'].value_counts())
print("\nPclass:\n", df['Pclass'].value_counts())
print("\nEmbarked:\n", df['Embarked'].value_counts())
print("\nSurvived:\n", df['Survived'].value_counts())

# Grouped analysis
print("\nAverage Age by Class:")
print(df.groupby('Pclass')['Age'].mean())

print("\nSurvival Rate by Gender:")
print(df.groupby('Sex')['Survived'].mean())

# 4. Visualization Tasks
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title("Distribution of Age")
plt.show()

sns.countplot(data=df, x='Sex')
plt.title("Count by Gender")
plt.show()

sns.barplot(data=df, x='Sex', y='Survived')
plt.title("Survival Rate by Gender")
plt.show()

sns.barplot(data=df, x='Pclass', y='Survived')
plt.title("Survival by Class")
plt.show()

sns.boxplot(data=df, x='Survived', y='Age')
plt.title("Age vs Survival")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 5. Stretch Activity: Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("\nSurvival by Family Size:")
print(df.groupby('FamilySize')['Survived'].mean())

sns.violinplot(data=df, x='Survived', y='FamilySize')
plt.title("Survival by Family Size")
plt.show()


# second program
# Customer Segmentation using Descriptive Analytics - Simple Version
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_excel("Online_Retail.xlsx")

print("First 5 Rows:")
print(df.head())

# 2. Data Cleaning
df = df.dropna(subset=['CustomerID'])           # Remove missing CustomerID
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]  # Remove negative or zero

# 3. Descriptive Statistics
print("\nQuantity Mean:", df['Quantity'].mean())
print("Quantity Median:", df['Quantity'].median())
print("Quantity Mode:", df['Quantity'].mode()[0])
print("Quantity Std Dev:", df['Quantity'].std())

print("\nUnitPrice Mean:", df['UnitPrice'].mean())
print("UnitPrice Median:", df['UnitPrice'].median())
print("UnitPrice Mode:", df['UnitPrice'].mode()[0])
print("UnitPrice Std Dev:", df['UnitPrice'].std())

# Total spending per customer
df['TotalSales'] = df['Quantity'] * df['UnitPrice']
customer_spend = df.groupby('CustomerID')['TotalSales'].sum().sort_values(ascending=False)
print("\nTotal Spend per Customer:\n", customer_spend.head())

# 4. Distribution Analysis
plt.figure(figsize=(10, 5))
sns.histplot(df['Quantity'], bins=50, kde=True)
plt.title("Distribution of Quantity")
plt.show()

sns.boxplot(df['UnitPrice'])
plt.title("Boxplot of Unit Price")
plt.show()

sns.kdeplot(df['TotalSales'])
plt.title("KDE Plot of Total Sales")
plt.show()

# 5. Group-Wise Analysis
print("\nTop Countries by Total Sales:")
print(df.groupby('Country')['TotalSales'].sum().sort_values(ascending=False).head())

print("\nTop Selling Products:")
print(df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head())

# 6. Visualization
top10_customers = customer_spend.head(10)
top10_customers.plot(kind='bar', title="Top 10 Customers by Total Spend", figsize=(10, 5))
plt.show()

country_sales = df.groupby('Country')['TotalSales'].sum()
country_sales.plot(kind='pie', autopct='%1.1f%%', title="Country-wise Sales Distribution", figsize=(8, 8))
plt.ylabel('')
plt.show()

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.to_period('M')
monthly_sales = df.groupby('Month')['TotalSales'].sum()
monthly_sales.plot(kind='line', title="Monthly Sales Trend", marker='o', figsize=(10, 5))
plt.show()

# 7. Insights
print("\nHigh-Value Customers:")
print(customer_spend.head(5))

print("\nPeak Purchasing Periods:")
print(monthly_sales.sort_values(ascending=False).head())

print("\nUnderperforming Products:")
print(df.groupby('StockCode')['Quantity'].sum().sort_values().head(5))


# .......# third program

# -----------------------------
# EXERCISE 2: DATA PREPROCESSING
# -----------------------------

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import zscore
from sklearn.feature_selection import VarianceThreshold

# 2. CREATE SAMPLE DATASET
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, np.nan, 35, 45, 29],
    'Gender': ['F', 'M', 'M', np.nan, 'F'],
    'Income': [50000, 60000, 80000, 120000, np.nan],
    'Loan_Status': ['Y', 'N', 'Y', 'N', 'Y']
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df, "\n")

# --------------------------------------------------
# 3. HANDLING MISSING VALUES
# --------------------------------------------------

print("Missing Values Count:\n", df.isnull().sum(), "\n")

# Fill missing Age with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing Gender with mode
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

# Fill missing Income with median
df['Income'].fillna(df['Income'].median(), inplace=True)

print("After Handling Missing Values:\n", df, "\n")

# --------------------------------------------------
# 4. ENCODING CATEGORICAL VARIABLES
# --------------------------------------------------

le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])    # F=0, M=1
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])  # N=0, Y=1

print("After Encoding Categorical Variables:\n", df, "\n")

# --------------------------------------------------
# 5. FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])

print("After Feature Scaling (Standardization):\n", df, "\n")

# --------------------------------------------------
# 6. OUTLIER DETECTION & TREATMENT
# --------------------------------------------------

z_scores = zscore(df[['Age', 'Income']])
outliers = (np.abs(z_scores) > 3).any(axis=1)
print("Detected Outliers:\n", df[outliers], "\n")

# Example of treating outliers (capping)
df['Income'] = np.where(df['Income'] > 2.5, 2.5, df['Income'])

# --------------------------------------------------
# 7. FEATURE SELECTION (Optional)
# --------------------------------------------------

selector = VarianceThreshold(threshold=0.1)
selected_features = selector.fit_transform(df[['Age', 'Income', 'Gender']])
print("Selected Features after Variance Threshold:\n", selected_features, "\n")

# --------------------------------------------------
# 8. FINAL PREPROCESSED DATA
# --------------------------------------------------

print("Final Preprocessed Data:\n", df)




# .........................................................

# fourth program
# ----------------------------------------------------------
# CA3301 - Machine Learning
# Lab Exercise 4: FIND-S Algorithm Implementation
# ----------------------------------------------------------

# Step 1: Define the training dataset
# Each row = [Sky, AirTemp, Humidity, Wind, Water, Forecast, EnjoySport]
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High',   'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Cool', 'Change', 'Yes']
]

# Step 2: Initialize hypothesis with the most specific (ϕ)
hypothesis = ['ϕ'] * (len(data[0]) - 1)   # 6 attributes
print("Initial Hypothesis:", hypothesis)

# Step 3: Find the first positive example and initialize hypothesis
for example in data:
    if example[-1] == 'Yes':          # check if class label = positive
        hypothesis = example[:-1]     # copy all attribute values except the label
        break                         # initialize once, then stop

print("\nAfter first positive example:\n", hypothesis)

# Step 4: Compare with remaining positive examples and generalize if needed
for example in data:
    if example[-1] == 'Yes':          # process only positive examples
        for i in range(len(hypothesis)):
            if hypothesis[i] != example[i]:
                hypothesis[i] = '?'   # generalize when mismatch found

# Step 5: Display the final hypothesis
print("\nFinal Hypothesis:", hypothesis)


# ..............................................................# fifth program

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a simple dataset
data = {
    'Hours_Studied': [1, 2, 2.5, 3, 3.5, 4, 5, 6, 6.5, 7, 8, 9],
    'Passed': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Step 2: Separate features (X) and target (y)
X = df[['Hours_Studied']]  # independent variable
y = df['Passed']            # dependent variable

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4: Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict for a new student
hours = 4.5
prob = model.predict_proba([[hours]])[0][1]
prediction = model.predict([[hours]])[0]
print(f"\nPredicted probability of passing if studied {hours} hours: {prob:.2f}")
print("Prediction:", "Pass" if prediction == 1 else "Fail")

# Step 8: Visualization (Logistic Curve)
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_probs = model.predict_proba(X_plot)[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_plot, y_probs, color='red', linewidth=2, label='Logistic Curve')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Hours Studied vs Probability of Passing')
plt.legend()
plt.grid(True)
plt.show()
# ..............................................................
# sixth program

# ==========================================
# Exercise 5b: Logistic Regression Example
# ==========================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------
# Step 1: Create Sample Dataset
# ------------------------------------------
data = {
    'Hours_Studied': [1, 2, 2.5, 3, 3.5, 4, 5, 6, 6.5, 7, 8, 9],
    'Passed': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
print("Sample Data:")
print(df)
print()

# ------------------------------------------
# Step 2: Prepare Features (X) and Labels (y)
# ------------------------------------------
X = df[['Hours_Studied']]  # Independent variable (2D)
y = df['Passed']            # Dependent variable (1D)

# ------------------------------------------
# Step 3: Split into Train and Test Sets
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ------------------------------------------
# Step 4: Train Logistic Regression Model
# ------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------------------------
# Step 5: Make Predictions
# ------------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------------
# Step 6: Evaluate the Model
# ------------------------------------------
print("✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------
# Step 7: Predict for New Data
# ------------------------------------------
hours = 4.5
prob = model.predict_proba([[hours]])[0][1]
prediction = model.predict([[hours]])[0]
print(f"\nPredicted Probability of Passing if studied {hours} hours: {prob:.2f}")
print("Prediction:", "🎓 Pass" if prediction == 1 else "❌ Fail")

# ------------------------------------------
# Step 8: Visualization (Optional)
# ------------------------------------------
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_probs = model.predict_proba(X_plot)[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_plot, y_probs, color='red', linewidth=2, label='Logistic Curve')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Hours Studied vs Probability of Passing')
plt.legend()
plt.grid(True)
plt.show()




