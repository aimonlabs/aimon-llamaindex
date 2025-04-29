import os
import json
import itertools
from aimon import Client
from llama_index.llms.openai import OpenAI
from llama_index.core.response import Response
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (VectorStoreIndex, Document,StorageContext, load_index_from_storage)
from evaluators import (AIMonEvaluator, 
                        HallucinationEvaluator, 
                        ConcisenessEvaluator,
                        CompletenessEvaluator, 
                        GuidelineEvaluator, 
                        ToxicityEvaluator, 
                        ContextRelevanceEvaluator
                    )

##  Utility for printing
def print_result(result, label=None):
    try:
        if label:
            print()
            print(f"--- {label} ---")
        if hasattr(result, "model_dump"):
            print(json.dumps(result.model_dump(), indent=2))
        elif hasattr(result, "__dict__"):
            print(json.dumps(result.__dict__, indent=2))
        else:
            print(str(result))
            print()
    except Exception as e:
        print(f"(Could not serialize result: {e})")
        print(str(result))

## Individual evaluators
def run_individual_evaluators(aimon_client, user_query, llm_response, user_instructions, task_definition, context):
    print("\n### Individual Evaluators\n")

    aimon_evaluator = AIMonEvaluator(aimon_client)
    aimon_result = aimon_evaluator.evaluate(user_query, llm_response, user_instructions, task_definition)
    print_result(aimon_result, "AIMon Evaluator [Complete]")

    hallucination_evaluator = HallucinationEvaluator(aimon_client)
    hallucination_result = hallucination_evaluator.evaluate(user_query, llm_response)
    print_result(hallucination_result, "Hallucination Evaluator")

    conciseness_evaluator = ConcisenessEvaluator(aimon_client)
    conciseness_result = conciseness_evaluator.evaluate(user_query, llm_response)
    print_result(conciseness_result, "Conciseness Evaluator")

    completeness_evaluator = CompletenessEvaluator(aimon_client)
    completeness_result = completeness_evaluator.evaluate(user_query, llm_response)
    print_result(completeness_result, "Completeness Evaluator")

    guideline_evaluator = GuidelineEvaluator(aimon_client)
    guideline_result = guideline_evaluator.evaluate(user_query, llm_response, user_instructions)
    print_result(guideline_result, "Guideline Adherence Evaluator")

    toxicity_evaluator = ToxicityEvaluator(aimon_client)
    toxicity_result = toxicity_evaluator.evaluate(user_query, llm_response)
    print_result(toxicity_result, "Toxicity Evaluator")

    context_relevance_evaluator = ContextRelevanceEvaluator(aimon_client)
    context_relevance_result = context_relevance_evaluator.evaluate(user_query, llm_response, context)
    print_result(context_relevance_result, "Context Relevance Evaluator")

## Pairwise combinations of evaluators
def run_pairwise_evaluations(aimon_client, user_query, llm_response, user_instructions, task_definition):
    detectors = [
        "hallucination", "instruction_adherence",
        "conciseness", "completeness",
        "toxicity", "context_relevance"
    ]

    detector_pairs = list(itertools.combinations(detectors, 2))
    print(f"\n### Pairwise Evaluator Runs ({len(detector_pairs)} combinations)\n")

    for pair in detector_pairs:
        name = " + ".join(pair)
        print(f"\n--- Evaluation: {name} ---")

        detector_config = {det: {"detector_name": "default"} for det in pair}

        evaluator = AIMonEvaluator(
            aimon_client,
            detector_configuration=detector_config
        )

        result = evaluator.evaluate(
            user_query=user_query,
            llamaindex_llm_response=llm_response,
            user_instructions=user_instructions,
            task_definition=task_definition
        )

        print_result(result, name)

## Main
def main():

    persist_dir = "index_storage"

    aimon_client = Client(auth_header=f"Bearer {os.getenv('AIMON_API_KEY')}")

    user_query = "What is the capital of France?"
    task_definition = "The domain is geography."
    user_instructions = [
        "Limit the response to under 100 words.",
        "The language should be English"
    ]

    context = """Paris is the capital of France.
    Tokyo is the capital of Japan.
    Canberra is the capital of Australia.
    Ottawa is the capital of Canada.
    Brasília is the capital of Brazil."""

    ## Load or Build Index
    if os.path.exists(persist_dir):
        print("Loading index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        print("Building index and saving to disk...")
        embedding_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)
        documents = [Document(text=context)]
        index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
        index.storage_context.persist(persist_dir=persist_dir)

    ## Build LLM and Query Engine
    llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        system_prompt="Please be professional and answer briefly."
    )
    llm.system_prompt += f" Please comply with these instructions: {user_instructions}."

    query_engine = index.as_query_engine(llm=llm)
    
    ## Get LLM response
    llm_response = query_engine.query("What is the capital of Australia?")

    ## Run Evaluations

    run_individual_evaluators(
        aimon_client=aimon_client,
        user_query=user_query,
        llm_response=llm_response,
        user_instructions=user_instructions,
        task_definition=task_definition,
        context=context
    )

    run_pairwise_evaluations(
        aimon_client=aimon_client,
        user_query=user_query,
        llm_response=llm_response,
        user_instructions=user_instructions,
        task_definition=task_definition
    )

if __name__ == "__main__":
    main()
