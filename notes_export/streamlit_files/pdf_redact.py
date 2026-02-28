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


pdf_file = "financial_document.pdf"
regex_pattern = r'\$[0-9]+,?[0-9]*'  
output_file = "redacted_output.pdf"

regex_redactor(pdf_file, regex_pattern, output_file)