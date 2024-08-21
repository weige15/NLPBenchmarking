# Import Document from lanchain
import os
from typing import Any, Optional, Sequence, List, Iterable

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator

from ..utils import create_logger
import pdb
logger = create_logger(__name__)

class LimitedSplitter(CharacterTextSplitter):
    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            # Wif we are about to overstep chunk size
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    # pdb.set_trace()
                    # logger.warning(
                    #     f"Created a chunk of size {total}, "
                    #     f"which is longer than the specified {self._chunk_size}"
                    # )
                    total = 0
                    current_doc = [] 
                    continue
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        minus_amnt = self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        total -= minus_amnt
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs



def generate_dataset(
    api_key: str,
    document_names: Sequence[str],
    test_size: int,
    generation_distributions: Optional[dict[Any, float]] = None,
):
    # text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50,separator=".")
    text_splitter = LimitedSplitter(chunk_size=300, chunk_overlap=50,separator=".")
    documents = []
    for dn in document_names:
        if not os.path.exists(dn):
            continue
        loader = TextLoader(dn)
        documents += loader.load()

    chunks = text_splitter.split_documents(documents)

    generation_distributions = generation_distributions or {
        simple: 0.5,
        reasoning: 0.25,
        multi_context: 0.25,
    }
    os.environ["OPENAI_API_KEY"] = api_key
    generator = TestsetGenerator.with_openai()
    testset = generator.generate_with_langchain_docs(
        chunks,
        test_size=test_size,
        distributions=generation_distributions,
    )
    return testset
