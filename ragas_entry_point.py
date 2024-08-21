import os
from argparse import ArgumentParser
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import Dataset
from langchain_core.prompts import PromptTemplate
from matplotlib.colors import LinearSegmentedColormap
from ragas import evaluate
from ragas.evaluation import Result
from ragas.metrics import (
    answer_relevancy,  
    context_precision,
    context_recall,
    # context_relevancy, Not usefule metric, deprecated see https://github.com/explodinggradients/ragas/pull/1111
    faithfulness,
)
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestDataset

from nlpbench.data.generation import generate_dataset
from nlpbench.models_for_eval import GPT35, ModelEvalInterface
from nlpbench.rags.generation import get_retriever
from nlpbench.utils import create_logger


def get_arguments():
    ap = ArgumentParser()
    ap.add_argument("--doc_locations", default="./data/docs/", type=str)

    return ap.parse_args()
    


def evaluate_model(model: ModelEvalInterface, test_df: pd.DataFrame):
    ## Get answers by conditioning the models on context retrieved from RAG
    data = {"question": [], "answer": [], "contexts": []}
    # Ensure that row["question"]  and row["context"] are strings
    assert isinstance(test_df["question"][0], str), "question must be a string"
    assert isinstance(test_df["context"][0], str), "context must be a string"

    for idx, row in test_df.iterrows():
        data["question"].append(row["question"])
        data["answer"].append(model.evaluate(row["question"])) # type: ignore
        data["contexts"].append(model.retrieve_context(row["question"])) # type: ignore
    dataset = Dataset.from_dict(data)

    # Evaluate the dataset
    result = None
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[
                # context_relevancy,
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            raise_exceptions=False,
        )
    except Exception as e:
        logger.error(e)
        logger.error("Evaluation failed, will exit")
        exit(1)

    assert result is Result, "Evaluation failed"

    result_df = result.to_pandas()
    result_df.to_csv("./data/initial_experiments/result.csv")

    # Show Results
    heatmap_data = result_df[
        [
            "context_relevancy",
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
        ]
    ]
    cmap = LinearSegmentedColormap.from_list("green_red", ["red", "green"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=0.5, cmap=cmap)
    plt.yticks(
        ticks=range(len(result_df["question"])),
        labels=result_df["question"],
        rotation=0,
    )
    plt.show()


def main():
    args = get_arguments()
    ### Look for all available documents in path 
    files_to_encode = []
    for root, dirs, files in os.walk(args.doc_locations):
        for file in files:
            files_to_encode.append(os.path.join(root, file))
            logger.info(f"Found file {file}. Will encode")


    ### Create the list of chunks to be used for generation
    logger.info(f"Generating dataset with {len(files_to_encode)}")
    testset: TestDataset = generate_dataset(
        api_key=os.environ["OPENAI_API_KEY"],
        document_names=files_to_encode,
        test_size=100,
        generation_distributions={
            simple: 0.5,
            reasoning: 0.25,
            multi_context: 0.25,
        },
    )

    ### Create RAG to supplement Model
    template = """ Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    logger.info("Creating RAG")
    vector_store_retriever = get_retriever(testset)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create Models to Evaluate
    models: Dict[str, ModelEvalInterface] = {
        "gpt-3.5-turbo": GPT35(
            prompt_template=prompt, vector_store_retriever=vector_store_retriever
        )
        # TODO: Add more models
        # "langchain" : ...
    }

    ### Start Generation Evaluation Data
    logger.info("Starting Evaluation")
    questions = testset.to_pandas()["question"].to_list()
    ground_truth = testset.to_pandas()["ground_truth"].to_list()
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}")
        evaluate_model(model, testset.to_pandas())


if __name__ == "__main__":
    # Make it so that logger for langchain is piped to debug
    logger = create_logger("ragas_entry_point")
    main()
