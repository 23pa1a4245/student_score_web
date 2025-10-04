import pandas as pd
import joblib
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '../models/trained_model.pkl')
model = joblib.load(model_path)

# Load test dataset
test_data_path = os.path.join(os.path.dirname(__file__), '../data/test_data.csv')
test_data = pd.read_csv(test_data_path)

# --- FIX: Select only the features used in training, in the same order ---
X_test = test_data[['WritingScore', 'ReadingScore', 'MathScore']]

# Predict overall scores
predictions = model.predict(X_test)

# Save predictions
output = test_data.copy()
output['predicted_overall'] = predictions

output_path = os.path.join(os.path.dirname(__file__), '../data/test_predictions.csv')
output.to_csv(output_path, index=False)

print(f"✅ Predictions saved at {output_path}")