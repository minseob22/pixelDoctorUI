#pdf 읽어서 랭체인 doc 형태로 변환
from langchain_community.document_loaders import (
    PyPDFLoader
)

def load_pdf(pdf_path):
    loader=PyPDFLoader(pdf_path)
    
    documents=loader.load()
    return documents