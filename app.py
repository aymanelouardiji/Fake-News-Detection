from __future__ import annotations

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Fake News Detection", page_icon="N", layout="centered")
st.title("Fake News Detection")
st.caption("Machine learning and deep learning classification of news content.")

user_text = st.text_area("Enter a news article or claim:", height=220)

if st.button("Analyze", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text before analysis.")
    else:
        try:
            response = requests.post(API_URL, json={"text": user_text}, timeout=30)
            response.raise_for_status()
            payload = response.json()
            label = payload["label"]
            if label == "FAKE":
                st.error(f"Prediction: {label}")
            else:
                st.success(f"Prediction: {label}")

            col1, col2 = st.columns(2)
            col1.metric("P(FAKE)", payload["probability_fake"])
            col2.metric("P(REAL)", payload["probability_real"])
            st.caption(f"Served by model: {payload['model_name']}")
        except requests.RequestException as exc:
            st.error(
                "FastAPI service is unreachable. Start it with: "
                "`uvicorn api:app --reload`"
            )
            st.exception(exc)

