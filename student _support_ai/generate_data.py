import numpy as np
import pandas as pd

# Reproducible random data
rng = np.random.default_rng(42)

# Number of students
n = 1500

# Generate student information
attendance = np.clip(
    rng.normal(76, 12, n),
    35,
    100
)

marks = np.clip(
    rng.normal(68, 14, n),
    25,
    100
)

backlogs = np.clip(
    rng.poisson(0.8, n),
    0,
    6
)

lms = np.clip(
    rng.normal(70, 18, n),
    5,
    100
)

previous_gpa = np.clip(
    rng.normal(7.0, 1.2, n),
    3,
    10
)

assignments = np.clip(
    rng.normal(74, 16, n),
    10,
    100
)

# Create dataframe
df = pd.DataFrame({
    "student_id": [f"S{1001+i}" for i in range(n)],

    "attendance_pct": attendance,

    "semester_marks_pct": marks,

    "backlogs": backlogs,

    "lms_engagement_pct": lms,

    "previous_gpa": previous_gpa,

    "assignment_completion_pct": assignments
})


# ---------------------------------------------------
# ADD SOME MISSING VALUES
# ---------------------------------------------------

missing_rates = {
    "attendance_pct": 0.04,
    "semester_marks_pct": 0.05,
    "lms_engagement_pct": 0.04,
    "previous_gpa": 0.03,
    "assignment_completion_pct": 0.04
}

for column, rate in missing_rates.items():

    number_missing = int(n * rate)

    indexes = rng.choice(
        n,
        number_missing,
        replace=False
    )

    df.loc[indexes, column] = np.nan


# ---------------------------------------------------
# CREATE DEMONSTRATION TARGET
# ---------------------------------------------------

risk_score = (

    0.035 *
    (75 - df["attendance_pct"].fillna(76))

    +

    0.030 *
    (65 - df["semester_marks_pct"].fillna(68))

    +

    0.75 *
    df["backlogs"]

    +

    0.020 *
    (65 - df["lms_engagement_pct"].fillna(70))

    +

    0.60 *
    (6.8 - df["previous_gpa"].fillna(7.0))

    +

    0.018 *
    (70 - df["assignment_completion_pct"].fillna(74))

    +

    rng.normal(0, 0.8, n)
)


# Top 22% approximately become positive cases
threshold = np.quantile(
    risk_score,
    0.78
)


df["support_needed_next_period"] = (
    risk_score >= threshold
).astype(int)


# ---------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------

df.to_csv(
    "data/student_support_demo.csv",
    index=False
)


print("Dataset created successfully!")

print("\nFirst 5 students:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(
    df["support_needed_next_period"]
    .value_counts()
)

print("\nDataset saved to:")
print("data/student_support_demo.csv")