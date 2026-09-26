import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib, os

def clean(df):
    df = df.drop_duplicates()
    df = df.drop(columns=["LoanID"])
    return df

def add_features(df):
    df["LoanToIncome"] = df["LoanAmount"] / df["Income"]
    for col in ["HasMortgage", "HasDependents", "HasCoSigner"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, columns=["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"], drop_first=True)
    return df

def split_and_scale(df, target_col="Default"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include="number").columns
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    os.makedirs("processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    X_train[target_col] = y_train.values
    X_test[target_col] = y_test.values
    X_train.to_csv("processed/train.csv", index=False)
    X_test.to_csv("processed/test.csv", index=False)
    joblib.dump(scaler, "models/preprocessor.pkl")
    print("Done! Saved train.csv and test.csv")

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Loan_Project\ingested.csv")
    df = clean(df)
    df = add_features(df)
    split_and_scale(df)
