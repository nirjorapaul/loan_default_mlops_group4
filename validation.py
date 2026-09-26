import pandas as pd
import json, os

def validate(df, target_col="Default"):
    report = {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),
        "negative_income": int((df["Income"] < 0).sum()),
        "target_default_rate": float(df[target_col].mean()),
    }
    return report

if __name__ == "__main__":
    df = pd.read_csv("ingested.csv")
    report = validate(df)
    os.makedirs("reports", exist_ok=True)
    with open("reports/data_quality.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
