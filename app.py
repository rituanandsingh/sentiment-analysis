import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ARTIFACTS = Path(__file__).parent / "Artifacts"
MODEL_PATH = ARTIFACTS / "BiGRU_Modle.keras"   # file name exactly as saved in the notebook
TOKENIZER_PATH = ARTIFACTS / "tokenizer.pkl"
MAX_LEN = 50  # must match pad_sequences(maxlen=50) used in training

# Same order as the dataset's label ids: 0..5
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOJI = {"sadness": "😢", "joy": "😄", "love": "❤️", "anger": "😠", "fear": "😨", "surprise": "😲"}

EXAMPLES = {
    "Joy": "I feel so happy and grateful for everything today.",
    "Sadness": "I feel so alone and hopeless today.",
    "Anger": "I am furious that they cancelled the trip at the last minute.",
    "Fear": "I feel terrified when walking down dark alleyways alone.",
}

st.set_page_config(page_title="Emotion Classifier", page_icon="🎭", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict(text: str) -> np.ndarray:
    model, tokenizer = load_artifacts()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return model.predict(padded, verbose=0)[0]


st.title("🎭 Emotion Classifier")
st.write(
    "Type a sentence and a Bi-directional GRU model will predict which of six emotions it expresses: "
    "sadness, joy, love, anger, fear or surprise."
)

# Example buttons fill the text box via session state
if "text" not in st.session_state:
    st.session_state.text = ""

cols = st.columns(len(EXAMPLES))
for col, (name, sample) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.text = sample

text = st.text_area("Your text", key="text", height=120, placeholder="e.g. I feel so happy today!")

if st.button("Predict emotion", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        probs = predict(text)
        top = int(np.argmax(probs))
        label = LABELS[top]

        st.divider()
        st.subheader(f"{EMOJI[label]}  {label.capitalize()}")
        st.caption(f"Confidence: {probs[top]:.1%}")

        chart_df = pd.DataFrame({"Probability": probs}, index=[l.capitalize() for l in LABELS])
        st.bar_chart(chart_df)

        if len(text.split()) > MAX_LEN:
            st.info(f"Only the first {MAX_LEN} words are used by the model.")

st.caption("Trained on the dair-ai/emotion dataset (short English tweets). It works best on first-person sentences like 'I feel ...'.")
