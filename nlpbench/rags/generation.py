from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


def get_retriever(chunks):
    """
    TODO: Too simple for its own function but perhaps can be expanded upon
    """
    embedding = OpenAIEmbeddings()

    # Create a Chroma vector store
    vectorstore = Chroma.from_documents(chunks, embedding)
    retriever = vectorstore.as_retriever()
    return retriever
