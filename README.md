# sarcasm-detector
A Streamlit app that detects sarcasm using distilBERT

# 🤖 Sarcasm Detector

A stylish and sassy web app that detects sarcasm in sentences using a fine-tuned DistilBERT model — built with ❤️ using Streamlit, Transformers, and TensorFlow.

## 🧠 What It Does

This app takes any English sentence and predicts whether it's sarcastic or not, using the power of BERT (specifically `DistilBERT`). 

Whether you're decoding passive-aggressive tweets or just testing out your tone, this app has your back. 😏


## 🛠️ How to Run the App Locally

###1. Clone the repo

git clone [https://github.com/yourusername/sarcasm-detector.git]
cd sarcasm-detector
###2. Install the dependencies
pip install -r requirements.txt

###3. Run it with Streamlit
streamlit run sarcasm.py
Make sure all the model/tokenizer files are in the same directory as sarcasm.py.

🧾 Project Files:
File	                    Description
sarcasm.py	              Main Streamlit app
tf_model.h5	              Trained model weights
config.json	              DistilBERT config
tokenizer_config.json, 
tokenizer.json,           Tokenizer files
vocab.txt.
requirements.txt	        Python dependencies

## Model File
You can download the trained model from [https://drive.google.com/file/d/1GR5nyfdj2FADRfkNteMXGUFpamUJ82ft/view?usp=sharing]

