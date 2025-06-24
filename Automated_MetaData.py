# streamlit_app.py - Enhanced UI for AutoMeta

import streamlit as st
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import yaml

from transformers import pipeline
from keybert import KeyBERT
import fitz  # PyMuPDF
import docx
import pytesseract
from pdf2image import convert_from_path

# Load Models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    kw_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")
    return summarizer, kw_model

summarizer, kw_model = load_models()

# Extract Text Functions
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc).strip()

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs).strip()

def extract_text_via_ocr(path):
    images = convert_from_path(path)
    return "\n".join(pytesseract.image_to_string(img) for img in images).strip()

# Chunking-based Summarization

def summarize_long_text(text, chunk_size=1000, max_chunks=5):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks[:max_chunks]:
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(result[0]['summary_text'])
    return " ".join(summaries)

# Metadata Generator
def generate_metadata(text, doc_type="Unknown"):
    # Generate summary
    summary = summarize_long_text(text)
    
    # Generate title (via summarizer)
    title_prompt = "Generate a title for the following document: " + text[:1000]
    title_result = summarizer(title_prompt, max_length=15, min_length=4, do_sample=False)
    title = title_result[0]["summary_text"].strip().upper()
    
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=8
    )
    return {
        "TITLE": title,
        "SUMMARY": summary,
        "KEYWORDS": [kw[0] for kw in keywords],
        "DOCUMENT_TYPE": doc_type,
        "WORD_COUNT": len(text.split())
    }

# UI Layout
st.set_page_config(page_title="AutoMeta Metadata Generator", layout="wide")

with st.container():
    st.markdown("""
        <h1 style='text-align: center; color: #4A6FA5;'>📄 AutoMeta</h1>
        <h4 style='text-align: center; color: #606060;'>AI-Powered Smart Metadata Generator for PDFs, DOCX, and TXT</h4>
        <hr style='margin-top:0;'>
    """, unsafe_allow_html=True)

uploaded = st.file_uploader(" Upload Document", type=["pdf", "docx", "txt"], help="Supported formats: PDF, DOCX, TXT")

if uploaded:
    with NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp_file:
        tmp_file.write(uploaded.read())
        tmp_path = tmp_file.name

    ext = Path(uploaded.name).suffix.lower()
    text, doc_type = "", "Unknown"

    if ext == ".pdf":
        text = extract_text_from_pdf(tmp_path)
        if not text:
            text = extract_text_via_ocr(tmp_path)
            doc_type = "Scanned PDF"
        else:
            doc_type = "PDF"

    elif ext == ".docx":
        text = extract_text_from_docx(tmp_path)
        doc_type = "DOCX"

    elif ext == ".txt":
        text = Path(tmp_path).read_text(encoding="utf-8")
        doc_type = "TXT"

    if text:
        st.success(" Document processed successfully.")
        metadata = generate_metadata(text, doc_type)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 style='color: black; font-weight: bold;'> JSON METADATA</h3>", unsafe_allow_html=True)
            st.json(metadata)
            json_str = json.dumps(metadata, indent=2)
            st.download_button(
                label="⬇DOWNLOAD JSON",
                data=json_str,
                file_name="metadata.json",
                mime="application/json"
            )

        with col2:
            st.markdown("<h3 style='color: black; font-weight: bold;'> YAML METADATA</h3>", unsafe_allow_html=True)
            st.code(yaml.dump(metadata, sort_keys=False), language="yaml")
            
        with st.expander(" Full Text Extracted"):
            st.text_area("Extracted Text", text, height=200)
    else:
        st.error(" Could not extract text from the file.")

else:
    st.info("Please upload a document to begin.")
