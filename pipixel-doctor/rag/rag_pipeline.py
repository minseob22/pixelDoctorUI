from rag.retriever import (
    retrieve_relevant_chunks
)

def build_rag_context(
    vector_store,
    semantic_data
):
    finding=semantic_data["finding"]
    query = f"""
    radiomics
    {finding.get("shape")}
    {finding.get("texture_complexity")}
    {finding.get("internal_heterogeneity")}
    tumor texture analysis
    CT imaging
    """
    results=retrieve_relevant_chunks(
        vector_store,
        query,
        k=3
    )
    
    context=""
    
    for i , doc in enumerate(results):
        context==(
            
         f"\n[paper chunk{i+1}]\n"   
        )
        
        context += doc.page_content
        
        context += "\n"
        
    return context