import streamlit as st
import pymupdf
import re

def regex_redactor(pdf_path, pattern_string, output_path):
    doc = pymupdf.open(pdf_path)
    compiled_pattern = re.compile(pattern_string)
    matches_by_page = {}
    for page_num, page in enumerate(doc):
        word_list = page.get_text("words", sort=True) 
        matches = [
            pymupdf.Rect(w[:4]) 
            for w in word_list 
            if compiled_pattern.search(w[4])
        ]
        if matches:
            matches_by_page[page_num] = matches
            for inst in matches:
                page.add_redact_annot(inst, "REDACTED", fontname="helv", fontsize=11)
        page.apply_redactions()         
    doc.save(output_path)
    return f"{len(matches_by_page[page_num])} items were redacted on {len(matches_by_page)} pages."


st.title("PDF Regex Redactor")

st.write("This app redacts text in a PDF file based on a user-specified regex pattern.")

st.sidebar.header("Upload PDF and Specify Regex Pattern")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
regex_pattern = st.sidebar.text_input("Enter regex pattern to redact", r'\$[0-9]+,?[0-9]*')  # Default pattern to match dollar amounts

start_redaction = st.sidebar.button("Start Redaction")

if start_redaction and uploaded_file is not None:
    with open("temp_input.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    output_pdf_path = "redacted_output.pdf"
    result_message = regex_redactor("temp_input.pdf", regex_pattern, output_pdf_path)
    
    st.success("Redaction complete!")
    st.write(result_message)
    
    with open(output_pdf_path, "rb") as f:
        st.download_button(
            label="Download Redacted PDF",
            data=f,
            file_name="redacted_output.pdf",
            mime="application/pdf"
        )