import re
import pandas as pd
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    url_pattern = re.compile(r'https?://\S+')
    otp_pattern = re.compile(r'\b(otp|pin|password|cvv|one[- ]?time)\b', re.I)
    credential_pattern = re.compile(r'\b(login|verify|account|bank|secure|update|credentials)\b', re.I)
    df["contains_url"] = df["text"].apply(lambda x: bool(url_pattern.search(x)))
    df["contains_otp_request"] = df["text"].apply(lambda x: bool(otp_pattern.search(x)))
    df["contains_credential_terms"] = df["text"].apply(lambda x: bool(credential_pattern.search(x)))
    return df

if __name__ == "__main__":
    df = pd.read_csv("dataset/english_raw.csv")
    df = extract_features(df)
    df.to_csv("dataset/english_augmented.csv", index=False)
    print("✅ Features added → dataset/english_augmented.csv")
