import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("HousePricePrediction.csv")

# Data Preprocessing - Data cleaning
# Drop Id as it doesn't contribute in price
df.drop(['Id'], axis=1, inplace=True)
# Remove duplicate rows
df.drop_duplicates(inplace=True) 
# Replacing SalePrice empty values with their mean values
df['SalePrice'] = df['SalePrice'].fillna(
  df['SalePrice'].mean()) 
# Drop records with null values
df = df.dropna()

# Separate Features and Target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Find categorical columns
categorical_columns = X.select_dtypes(include=["object"]).columns

# Create Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# Create Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Train Model
pipeline.fit(X, y)

# Save Model
joblib.dump(pipeline, "model.pkl")

print("Model trained successfully!")
print("model.pkl has been created.")