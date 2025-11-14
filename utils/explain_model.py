import os
import numpy as np
import pandas as pd
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer
from typing import List, Tuple, Optional

model = None
tokenizer = None
device = "cpu"

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "phish_detector")
FALLBACK_MODEL = "xlm-roberta-base"
MAX_LEN = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer: Optional[XLMRobertaTokenizer] = None
model: Optional[XLMRobertaForSequenceClassification] = None


def load_model_for_language(lang_code: str = "en") -> Tuple[XLMRobertaTokenizer, XLMRobertaForSequenceClassification]:
    """
    Loads the fine-tuned model for English if available; for other languages loads base multilingual model.
    Returns tokenizer, model (moved to device).
    """
    global tokenizer, model
    if lang_code.lower().startswith("en") and os.path.isdir(MODEL_PATH):
        try:
            tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
            model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
            model.to(device)
            model.eval()
            print("✅ Loaded fine-tuned model from:", MODEL_PATH)
            return tokenizer, model
        except Exception as e:
            print("⚠️ Failed to load local fine-tuned model:", e)
    try:
        tokenizer = XLMRobertaTokenizer.from_pretrained(FALLBACK_MODEL)
        model = XLMRobertaForSequenceClassification.from_pretrained(FALLBACK_MODEL, num_labels=2)
        model.to(device)
        model.eval()
        print(f"🌍 Loaded fallback multilingual model: {FALLBACK_MODEL}")
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Failed to load fallback model ({FALLBACK_MODEL}): {e}")

try:
    tokenizer, model = load_model_for_language("en")
except Exception as e:
    print("❌ Model initialization failed at import:", e)
    tokenizer = None
    model = None

def set_model_objects(m, t, dev):
    """Attach model, tokenizer, and device from Streamlit app."""
    global model, tokenizer, device
    model = m
    tokenizer = t
    device = dev


def _ensure_model():
    """Raise an informative error if model/tokenizer not loaded."""
    if tokenizer is None or model is None:
        raise RuntimeError("Model/tokenizer not loaded. Call load_model_for_language() or check model files.")


def predict_proba(texts: List[str]) -> np.ndarray:
    """
    Accepts list of texts (or single-element list) and returns numpy array of shape (n_samples, 2)
    with probabilities for [ham, phishing].
    """
    _ensure_model()

    if isinstance(texts, str):
        texts = [texts]

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
    return probs  


def get_explanation_data(text: str, num_features: int = 6, num_samples: int = 50) -> pd.DataFrame:
    """
    Faster LIME explanation with reduced sampling.
    Default num_samples=50 (vs 500 default in LIME).
    Returns DataFrame with columns: word, weight, influence.
    """
    _ensure_model()

    explainer = LimeTextExplainer(class_names=["ham", "phishing"])

    def classifier_fn(list_of_texts: List[str]) -> np.ndarray:
        return predict_proba(list_of_texts)
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=classifier_fn,
        num_features=num_features,
        num_samples=num_samples, 
        labels=(1,)
    )
    items = exp.as_list(label=1)
    df = pd.DataFrame(items, columns=["word", "weight"])
    df["influence"] = np.where(df["weight"] > 0, "phishing", "ham")
    return df

if __name__ == "__main__":
    sample_text = "Your bank account has been locked. Click here to verify your details."
    print("Sample text:", sample_text)
    try:
        probs = predict_proba([sample_text])
        print("Probs:", probs)
        explanation_df = get_explanation_data(sample_text)
        print("\nTop contributing tokens:")
        print(explanation_df)
    except Exception as e:
        print("Error during demo:", e)
