import os
import pickle
import streamlit as st

from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(
    page_title="Twitter Sentiment Analysis (RAG)",
    page_icon="🐦",
    layout="centered"
)

# ==============================
# LOAD ML MODEL
# ==============================
@st.cache_resource
def load_ml_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

ml_model = load_ml_model()

# ==============================
# LOAD EMBEDDINGS
# ==============================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ==============================
# CREATE VECTOR DB (ONLY ONCE)
# ==============================
def create_db():
    loader = TextLoader("data.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="vector_db"
    )

# ==============================
# LOAD VECTOR DB (CHROMA)
# ==============================
@st.cache_resource
def load_db():
    if not os.path.exists("vector_db"):
        create_db()

    return Chroma(
        persist_directory="vector_db",
        embedding_function=embeddings
    )

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 2})

# ==============================
# LOAD LLM (FLAN-T5)
# ==============================
@st.cache_resource
def load_llm():
    gen_pipeline = pipeline(
        "text2text-generation",   # ✅ CORRECT FOR FLAN-T5
        model="google/flan-t5-small",
        max_new_tokens=120
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)

llm = load_llm()

# ==============================
# UI
# ==============================
st.title("🐦 Twitter Text Sentiment Analysis")
st.write("ML + RAG (LangChain + ChromaDB)")

tweet = st.text_input("✍️ Enter Tweet")

if st.button("Analyze"):
    if tweet.strip() == "":
        st.warning("Please enter text")
    else:
        # ------------------------------
        # ML Prediction
        # ------------------------------
        ml_pred = ml_model.predict([tweet])[0]
        ml_sentiment = "Positive" if ml_pred == 1 else "Negative"

        # ------------------------------
        # RAG Retrieval
        # ------------------------------
        docs = retriever.invoke(tweet)
        context = "\n".join(d.page_content for d in docs)

        # ------------------------------
        # Prompt
        # ------------------------------
        prompt = f"""
You are validating a sentiment classification.

ML Prediction: {ml_sentiment}

Similar labeled examples:
{context}

Tweet:
"{tweet}"

Return:
Sentiment:
Explanation:
Confidence:
"""

        final_result = llm.invoke(prompt)

        # ------------------------------
        # OUTPUT
        # ------------------------------
        st.subheader("📊 Result")
        st.write(final_result)


