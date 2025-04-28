import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import gdown
import os

# Download tf_model.h5 if not already present
if not os.path.exists("tf_model.h5"):
    url = "https://drive.google.com/uc?id=1GR5nyfdj2FADRfkNteMXGUFpamUJ82ft"
    gdown.download(url, "tf_model.h5", quiet=False)

# 1. Load tokenizer properly
tokenizer = DistilBertTokenizerFast.from_pretrained("./")

# 2. Build model (architecture first)
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# 3. Load your .h5 weights
model.load_weights("tf_model.h5")

# 4. Prediction function
def predict_sarcasm(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = model(inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1)  # Convert to probabilities
    sarcastic_score = probs[0][1].numpy()   # Probability of sarcastic class (1)
    return sarcastic_score

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #00ffe5;'>🔥 Sarcasm Detector 🔥</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #cccccc;'>Because sometimes... words are sharper than knives 😈</h5>", unsafe_allow_html=True)
st.markdown("---")

user_input = st.text_area("💬 Type something juicy...")

if st.button("👀 Detect Sarcasm"):
    if user_input.strip() == "":
        st.warning("🥺 please enter something...")
    else:
        sarcasm_probability = predict_sarcasm(user_input)
        
        st.write(f"*Sarcasm Probability:* {sarcasm_probability:.4f}")

        if sarcasm_probability > 0.5:
            st.error("🧠 Detected: Sarcastic! 😏")
        else:
            st.success("🌸 Detected: Not Sarcastic! 💖")
