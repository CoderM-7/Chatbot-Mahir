import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="ChatBot", layout="centered")
st.title(" Mahir ChatBot")

model_name = st.sidebar.selectbox(
    "Choose a Hugging Face Model",
    ["google/flan-t5-base", "google/flan-t5-large"]
)

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_model(model_name)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = ConversationChain(llm=llm, memory=st.session_state.memory)

user_input = st.chat_input("Type your message...")

if user_input:
    with st.spinner("Thinking..."):
        response = st.session_state.chat_chain.predict(input=user_input)

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", response))

if "history" in st.session_state:
    for sender, message in st.session_state.history:
        if sender == "You":
            st.markdown(f"** {sender}:** {message}")
        else:
            st.markdown(f"** {sender}:** {message}")