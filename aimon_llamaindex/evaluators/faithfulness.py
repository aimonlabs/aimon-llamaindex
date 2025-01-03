import logging
from aimon import Client
from typing import Optional, Any                                              
from .aimon_evaluator import AIMonEvaluator
from llama_index.core.evaluation import EvaluationResult

class FaithfulnessEvaluator(AIMonEvaluator):
    
    ## Set the AIMon Client in the constructor
    def __init__(self, aimon_client: Client, publish:bool=False) -> None:                  
        
        super().__init__(aimon_client, publish)

    ## Create AIMon Payload for the Faithfulness (Hallucination) Detector
    def create_payload(self, context, user_query, user_instructions, generated_text):
        
        aimon_payload = super().create_payload(context, user_query, user_instructions, generated_text)
        aimon_payload['config'] = { 'hallucination': {'detector_name': 'default'},}

        return aimon_payload


    ## Function to evaluate the LLM response for Faithfulness
    ## According to LlamaIndex docs, the "evaluate" method takes in query, contexts, response, and additional keyword arguments.

    def evaluate(self, user_query, user_instructions, llamaindex_llm_response, **kwargs:Any):
        
        context, response = self.extract_response_metadata(llamaindex_llm_response)

        aimon_payload = self.create_payload(context, user_query, user_instructions, response)
    
        detect_response = self.detect_aimon_response(aimon_payload)

        ## Create evaluation result        
        evaluation_result = EvaluationResult()
        
        evaluation_result.score = detect_response.hallucination['score'] 
        
        if evaluation_result.score <= 0.5:
            evaluation_result.passing = True
            evaluation_result.feedback = "The LLM response is not hallucinated."
        else:
            evaluation_result.passing = False
            evaluation_result.feedback = "The LLM response is hallucinated."

        return evaluation_result