import streamlit as st
from inference.logistic_infer import predict_logistic
from inference.vader_infer import predict_vader
from inference.roberta_infer import predict_roberta

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("Sentiment Analysis Using NLP")
st.write("Final Year Project")

text = st.text_area("Enter text to analyze sentiment")

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "VADER", "Fine-tuned RoBERTa", "Compare All"]
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        if model_choice == "Logistic Regression":
            st.success(predict_logistic(text))

        elif model_choice == "VADER":
            st.success(predict_vader(text))

        elif model_choice == "Fine-tuned RoBERTa":
            st.success(predict_roberta(text))

        else:
            st.subheader("Model Comparison")
            st.write("Logistic Regression:", predict_logistic(text))
            st.write("VADER:", predict_vader(text))
            st.write("RoBERTa:", predict_roberta(text))
