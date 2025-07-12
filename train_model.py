# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
df = pd.read_csv("CO2 Emissions_Canada.csv")

# Select features and target
X = df[['Fuel Type', 'Fuel Consumption Comb (L/100 km)']]
y = df['CO2 Emissions(g/km)']

# Define preprocessing (OneHot for Fuel Type)
preprocessor = ColumnTransformer(
    transformers=[
        ('fuel_type', OneHotEncoder(), ['Fuel Type'])
    ],
    remainder='passthrough'  # keep numeric column
)

# Build pipeline with preprocessing + model
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', LinearRegression())
])

# Train the pipeline
pipeline.fit(X, y)

# Save the trained pipeline model
with open('co2_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as co2_model.pkl")
