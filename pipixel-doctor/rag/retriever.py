# 쿼리랑 가장 유사한 논문 벡터검색으로 찾아옴.

def retrieve_relevant_chunks(
    vector_store,
    query,
    k=3
):
    
    results=vector_store.similarity_search(
        query,
        k=k
    )
    return results