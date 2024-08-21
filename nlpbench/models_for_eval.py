from abc import ABC, abstractmethod

from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_openai import ChatOpenAI
from typing import Sequence


class ModelEvalInterface(ABC):
    @abstractmethod
    def evaluate(self, query: str) -> str:
        pass
    @abstractmethod
    def retrieve_context(self, query: str) -> Sequence[str]:
        pass

class GPT35(ModelEvalInterface):
    def __init__(self, **kwargs):
        prompt_template_str = kwargs["prompt_template_str"]
        self.retriever = kwargs["retriever"]
        assert isinstance(prompt_template_str, str), "prompt_template must be a PromptTemplate"
        assert isinstance(self.retriever, Chroma), "retriever must be a Chroma"

        model = ChatOpenAI(name="gpt-3.5-turbo", temperature=0.0)
        prompt = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])
        self.gpt35_rag_chain: RunnableSerializable = (
            {"context": kwargs["vector_store_retriever"], "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        ChatOpenAI(name="gpt-3.5-turbo", temperature=0.0)
        pass
    def evaluate(self, query):
        return self.gpt35_rag_chain.invoke(query)

    def retrieve_context(self, query) -> Sequence[str]:
        return [
            doc.page_content for doc in self.retriever.get_relevant_documents(query)
        ]


# TODO: Create Langfuse Version
class LangFuse(ModelEvalInterface):
    def __init__(self, **kwargs):
        pass
    def evaluate(self, model, dataset):
        pass
    def retrieve_context(self, query):
        pass
