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
