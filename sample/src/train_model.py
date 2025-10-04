import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset (directly from your CSV)
data = pd.read_csv("data/data.csv")
# Features and target
X = data[['WritingScore', 'ReadingScore', 'MathScore']]
y = data['Overall']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "models/trained_model.pkl")

print("✅ Model trained and saved at models/trained_model.pkl")
