import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('students_data.csv')
print(df.columns)  # Show actual column names

# Use actual column names from CSV
X = df[['Hours_Studied']]
y = df['Results']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

new_hours = [[7.5]]
result = model.predict(new_hours)
print("Prediction for 7.5 hours studied:", "Pass" if result[0] == 1 else "Fail")
