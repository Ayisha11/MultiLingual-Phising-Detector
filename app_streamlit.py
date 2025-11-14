import importlib.util
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from googletrans import Translator  

spec = importlib.util.spec_from_file_location("explain_model", os.path.join("utils", "explain_model.py"))
explain_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(explain_model)

get_explanation_data = explain_model.get_explanation_data
predict_proba = explain_model.predict_proba

MODEL_PATH = "models/phish_detector"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model not found! Please train the model first using `train_model.py`.")
    st.stop()

@st.cache_resource
def load_model():
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
explain_model.set_model_objects(model, tokenizer, device)

st.set_page_config(page_title="Phishing Detection Dashboard", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
        body { background-color: #f7f9fb; }
        .big-font { font-size:25px !important; color:#003366; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛡️ Multilingual Phishing Email Detector with Explainable AI")
st.markdown("### Detect & Understand Phishing Messages in Multiple Languages")

input_text = st.text_area("✉️ Paste or type your email / message:", height=150)

col1, col2 = st.columns([1, 3])

with col1:
    language = st.selectbox("🌍 Language", ["English", "Tamil", "Hindi", "French", "Spanish", "German"])

with col2:
    if st.button("🔍 Analyze"):
        if input_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing message..."):
                probs = predict_proba([input_text])[0]
                phishing_prob = probs[1]
                label = "Phishing 🧠" if phishing_prob > 0.5 else "Safe ✅"

                st.markdown(f"## Prediction: **{label}**")
                st.progress(float(phishing_prob))
                st.write(f"Phishing probability: **{phishing_prob:.2f}**")
                st.write(f"Safe probability: **{probs[0]:.2f}**")

                st.divider()
                st.subheader("🔎 Why did the model predict this?")

                with st.spinner("🧩 Generating explanation... (this may take a few seconds)"):
                    exp_df = get_explanation_data(input_text)

                fig = px.bar(
                    exp_df.sort_values(by="weight", ascending=False),
                    x="word",
                    y="weight",
                    color="influence",
                    color_discrete_map={"phishing": "#ff6b6b", "ham": "#00b894"},
                    text="weight"
                )
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig.update_layout(
                    title="Word Importance for Classification",
                    title_font=dict(size=18, color="white"),
                    xaxis_title="Word",
                    yaxis_title="Influence Weight",
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font=dict(color="white"),
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                top_words = exp_df.head(3)["word"].tolist()
                phish_weight = exp_df[exp_df["influence"] == "phishing"]["weight"].sum()
                ham_weight = exp_df[exp_df["influence"] == "ham"]["weight"].sum()

                if abs(phish_weight) > abs(ham_weight):
                    reason = f"The model detected high-risk words such as {', '.join(top_words)}, which are commonly seen in scam or phishing emails."
                else:
                    reason = f"The message contains harmless words like {', '.join(top_words)}, suggesting it’s a normal communication."

                st.markdown("### 🧠 Model Interpretation Summary")
                st.info(reason)

                try:
                    translator = Translator()
                    detected_lang = translator.detect(input_text).lang
                    if detected_lang != "en":
                        reason_translated = translator.translate(reason, dest=detected_lang).text
                        st.markdown(f"**🌍 Explanation ({detected_lang.upper()}):** {reason_translated}")
                except Exception:
                    st.warning("🌐 Translation unavailable. Proceeding in English only.")

                st.markdown("### 🧩 Keyword Highlights")
                highlight_html = " ".join([
                    f"<span style='background-color: {'#ff7675' if row.influence == 'phishing' else '#55efc4'};"
                    f"padding:5px;border-radius:8px;margin:2px'>{row.word}</span>"
                    for _, row in exp_df.iterrows()
                ])
                st.markdown(highlight_html, unsafe_allow_html=True)

st.sidebar.header("📊 About This App")
st.sidebar.info(
    """
    **Features:**
    - Detects phishing messages across 6 languages  
    - Uses multilingual XLM-RoBERTa model  
    - LIME explainability to visualize decision logic  
    - Generates plain-language reasoning for predictions  
    """
)
