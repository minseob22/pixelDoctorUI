#논문 텍스트조각으로 나눔

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
    )

def chunck_documents(documents):
    
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks=splitter.split_documents(
        documents
    )
    return chunks