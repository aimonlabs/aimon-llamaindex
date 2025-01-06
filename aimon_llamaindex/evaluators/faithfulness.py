from typing import Any                                              
from aimon import Client
from .aimon_evaluator import AIMonEvaluator
from llama_index.core.evaluation import EvaluationResult

class FaithfulnessEvaluator(AIMonEvaluator):
    
    def __init__(self, aimon_client:Client, publish: bool = False, application_name:str = "ApplicationName", model_name:str = "ModelName") -> None:                  
        super().__init__(aimon_client, publish, application_name, model_name)

    def create_payload(self, context, user_query, user_instructions, generated_text):
        
        aimon_payload = super().create_payload(context, user_query, user_instructions, generated_text)
        aimon_payload['config'] = { 'hallucination': {'detector_name': 'default'},}

        return aimon_payload
    
    def evaluate(self, user_query, user_instructions, llamaindex_llm_response, **kwargs: Any):

        context, response = self.extract_response_metadata(llamaindex_llm_response)

        ## Using the overridden create_payload method
        aimon_payload = self.create_payload(context, user_query, user_instructions, response)
    
        detect_response = self.detect_aimon_response(aimon_payload)

        # Create evaluation result        
        evaluation_result = EvaluationResult()
        evaluation_result.score = detect_response.hallucination['score'] 
        evaluation_result.passing = bool(detect_response.hallucination['is_hallucinated'])

        if evaluation_result.passing == True:
            evaluation_result.feedback = "The LLM response is not hallucinated."
        else:
            evaluation_result.feedback = "The LLM response is hallucinated."

        return evaluation_result
    