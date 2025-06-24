
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import pytesseract
from PIL import Image
import os
import pdfplumber
import docx
from keybert import KeyBERT

# Initialize LLM
Checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(Checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(Checkpoint, device_map='auto', torch_dtype=torch.float32)

# Initialize Keyword Extractor
kw_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")

# Set page config
st.set_page_config(layout='wide', page_title="Universal Summarization App")

# Preprocess any file
def file_preprocessing(file_path, file_type):
    text = ""

    if file_type == "pdf":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''

    elif file_type == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])

    elif file_type == "txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

    elif file_type == "image":
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)

    else:
        raise ValueError("Unsupported file type")

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([text])
    return " ".join([doc.page_content for doc in docs])

# Summarization Pipeline
def llm_pipeline(filepath, file_type):
    input_text = file_preprocessing(filepath, file_type)

    summarizer = pipeline("summarization", model=base_model, tokenizer=tokenizer,
                          max_length=500, min_length=50, truncation=True)

    # Title
    title_prompt = "generate a title: " + input_text[:1000]
    title_result = summarizer(title_prompt, max_length=20, min_length=5, do_sample=False)
    title = title_result[0]['summary_text'].strip()

    # Summary
    result = summarizer(input_text)
    summary = result[0]['summary_text'].strip()

    # Keywords
    keywords = kw_model.extract_keywords(input_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=8)
    keyword_list = [kw[0] for kw in keywords]

    return {
        "title": title,
        "summary": summary,
        "keywords": keyword_list
    }

# Display PDF inline
@st.cache_data
def display_pdf(file):
    with open(file, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Main UI
def main():
    st.title("📄 Universal File Summarizer")

    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt", "jpg", "jpeg", "png"])

    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        if file_ext in ["jpg", "jpeg", "png"]:
            file_type = "image"
        elif file_ext == "pdf":
            file_type = "pdf"
        elif file_ext == "docx":
            file_type = "docx"
        elif file_ext == "txt":
            file_type = "txt"
        else:
            st.error("Unsupported file type!")
            return

        # Save uploaded file
        filepath = os.path.join("data", uploaded_file.name)
        os.makedirs("data", exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.read())

        if st.button("Summarize"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original File View")
                if file_type == "image":
                    st.image(filepath, use_column_width=True)
                elif file_type == "pdf":
                    display_pdf(filepath)
                else:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        st.text(f.read()[:2000])  # Show first part only

            with col2:
                st.subheader("AI Summary")
                result = llm_pipeline(filepath, file_type)
                st.markdown(f"**Title:** {result['title']}")
                st.success(result['summary'])
                st.markdown("**Keywords:**")
                st.write(", ".join(result['keywords']))

if __name__ == "__main__":
    main()
