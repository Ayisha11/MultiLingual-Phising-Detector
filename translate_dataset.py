import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

languages = {
    "ta": "Tamil",
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "de": "German"
}

df = pd.read_csv("dataset/english_augmented.csv")
translated_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    base_text = row["text"]
    label = row["label"]
    attack_type = row["attack_type"]
    email_id = row["id"]

    # keep original
    translated_rows.append({
        "id": f"{email_id}_en",
        "text": base_text,
        "language": "en",
        "label": label,
        "attack_type": attack_type
    })

    # translations
    for code, lang in languages.items():
        try:
            translated_text = GoogleTranslator(source="en", target=code).translate(base_text)
            translated_rows.append({
                "id": f"{email_id}_{code}",
                "text": translated_text,
                "language": code,
                "label": label,
                "attack_type": attack_type
            })
        except Exception as e:
            print(f"⚠️ Could not translate {email_id} to {lang}: {e}")

final_df = pd.DataFrame(translated_rows)
final_df.to_csv("dataset/multilingual_emails.csv", index=False, encoding="utf-8")
print("✅ Multilingual dataset saved → dataset/multilingual_emails.csv")
