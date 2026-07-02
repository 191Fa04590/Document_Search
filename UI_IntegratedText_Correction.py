import streamlit as st
import numpy as np
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Document Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Document Search")
st.write("Enter your query and get the most relevant document with AI generated response.")

# -------------------------------
# Static Documents
# -------------------------------
documents = [
    "TCS employees have a relax work environment.",
    "Accenture employees are hghly dedicated and work efficiently.",
    "Capgemini employees work very hard and are dedicated to their jobs.",
    "Cognizant is a multinatonal IT company with offices across India.",
    "AWC is flexlbe work model with better compensastion."
]

# -------------------------------
# Load Models (Cached)
# -------------------------------
@st.cache_resource
def load_models():

    grammar_pipeline = pipeline(
        "text2text-generation",
        model="vennify/t5-base-grammar-correction"
    )

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation"
    )

    chat_model = ChatHuggingFace(llm=llm)

    return grammar_pipeline, embedding, chat_model


grammar_pipeline, embedding, model = load_models()

# -------------------------------
# User Input
# -------------------------------
query = st.text_input(
    "Enter your Query",
    placeholder="Example: Which company's employees work hard?"
)

# -------------------------------
# Button
# -------------------------------
if st.button("Generate Data", use_container_width=True):

    if query.strip() == "":
        st.warning("Please enter a query.")
        st.stop()

    with st.spinner("Processing..."):

        # Grammar Correction
        corrected_sentences = []

        for text in documents:
            prompt = f"grammar: {text}"

            result = grammar_pipeline(
                prompt,
                num_return_sequences=1
            )

            corrected_sentences.append(
                result[0]["generated_text"]
            )

        # Embeddings
        user_vector = embedding.embed_query(query)

        document_vectors = embedding.embed_documents(
            corrected_sentences
        )

        similarity = cosine_similarity(
            [user_vector],
            document_vectors
        )

        best_index = np.argmax(similarity)

        best_document = corrected_sentences[best_index]

        score = similarity[0][best_index]

        # LLM Response
        response = model.invoke(best_document)

    # -------------------------------
    # Output
    # -------------------------------

    st.success("Relevant document found!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Relevant Document")
        st.info(best_document)

    with col2:
        st.subheader("Similarity Score")
        st.metric(
            label="Cosine Similarity",
            value=f"{score:.4f}"
        )

    st.divider()

    st.subheader("AI Generated Response")

    st.write(response.content)