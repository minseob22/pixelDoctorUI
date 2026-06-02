# rag/vector_store.py
#논문 chunk를 embedding 벡터로 변환하고 FAISS 벡터 DB로 저장/로드하는 모듈.


from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import (
    FAISS
)

from config import OPENAI_API_KEY


# =========================================
# Embedding Model 생성
# =========================================

def get_embeddings():

    """
    OpenAI embedding 모델 생성.
    """

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY
    )

    return embeddings


# =========================================
# Vector Store 생성
# =========================================

def build_vector_store(chunks):

    """
    논문 chunk들을 embedding하여
    FAISS vector DB 생성.
    """

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# =========================================
# Vector Store 저장
# =========================================

def save_vector_store(
    vector_store,
    save_path="papers/embeddings/faiss_index"
):

    """
    생성된 FAISS vector DB를
    로컬 디스크에 저장.
    """

    vector_store.save_local(save_path)

    print(
        f"\n[INFO] Vector store saved to: "
        f"{save_path}"
    )


# =========================================
# Vector Store 로드
# =========================================

def load_vector_store(
    load_path="papers/embeddings/faiss_index"
):

    """
    저장된 FAISS vector DB를
    로컬 디스크에서 로드.
    """

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        load_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print(
        f"\n[INFO] Vector store loaded from: "
        f"{load_path}"
    )

    return vector_store