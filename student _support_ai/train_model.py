import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

DATA_PATH = "data/student_support_demo.csv"
MODEL_PATH = "models/student_support_model.joblib"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 2. Define target
# ---------------------------------------------------------

TARGET = "support_needed_next_period"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in the dataset."
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ---------------------------------------------------------
# 3. Remove ID-like columns
# ---------------------------------------------------------

# Student IDs should not be used as predictive features.
id_columns = [
    col for col in X.columns
    if col.lower() in ["id", "student_id", "studentid"]
]

if id_columns:
    print("\nRemoving ID columns:")
    print(id_columns)
    X = X.drop(columns=id_columns)


# ---------------------------------------------------------
# 4. Identify numeric and categorical columns
# ---------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ---------------------------------------------------------
# 5. Preprocessing
# ---------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# ---------------------------------------------------------
# 6. Create model
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1
)


# ---------------------------------------------------------
# 7. Create complete ML pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ---------------------------------------------------------
# 8. Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------------
# 9. Train model
# ---------------------------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# ---------------------------------------------------------
# 10. Evaluate model
# ---------------------------------------------------------

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Support Needed", "Support Needed"]
    )
)


# ---------------------------------------------------------
# 11. Save model
# ---------------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\n" + "=" * 50)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 50)

print(f"\nSaved to:")
print(MODEL_PATH)