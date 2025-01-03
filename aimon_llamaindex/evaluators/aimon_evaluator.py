import logging
from aimon import Client

class AIMonEvaluator:

    ## Constructor of the Base Class
    def __init__(self, aimon_client:Client, publish: bool = False) -> None:
 
        self.publish = publish
        self.client = aimon_client

    ## AIMon payload creation
    def create_payload(self, context, user_query, user_instructions, generated_text) -> dict:

        aimon_payload = {
            'context': context,
            'user_query': user_query,
            'generated_text': generated_text,
            'instructions': user_instructions,
        }

        aimon_payload['publish'] = self.publish

        ## Default configuration for all evaluators
        aimon_payload['config'] = { 'hallucination': {'detector_name': 'default'},
                                    'conciseness': {'detector_name': 'default'},
                                    'completeness': {'detector_name': 'default'},
                                    'instruction_adherence': {'detector_name': 'default'},
                                    'toxicity': {'detector_name': 'default'},
                                }

        if self.publish:
            aimon_payload['application_name'] = "Default Application Name"
            aimon_payload['model_name'] = "Default Model Name"

        return aimon_payload
    

    ## AIMon Detect
    def detect_aimon_response(self,aimon_payload):
        
        try:
            detect_response = self.client.inference.detect(body=[aimon_payload])
            # Check if the response is a list
            if isinstance(detect_response, list) and len(detect_response) > 0:
                detect_result = detect_response[0]
            elif isinstance(detect_response, dict):
                detect_result = detect_response  # Single dict response
            else:
                raise ValueError("Unexpected response format from detect API: {}".format(detect_response))
        except Exception as e:
                # Log the error and raise it
                print(f"Error during detection: {e}")
                raise
        
        return detect_result
    
    
    ## Function to extract metadata from the response

    def extract_response_metadata(self, llm_response):

        def get_source_docs(llm_response):
            contexts = []
            relevance_scores = []
            if hasattr(llm_response, 'source_nodes'):
                for node in llm_response.source_nodes:
                    if hasattr(node, 'node') and hasattr(node.node, 'text') and hasattr(node, 'score') and node.score is not None:
                        contexts.append(node.node.text)
                        relevance_scores.append(node.score)
                    elif hasattr(node, 'text') and hasattr(node, 'score') and node.score is not None:
                        contexts.append(node.text)
                        relevance_scores.append(node.score)
                    else:
                        logging.info("Node does not have required attributes.")
            else:
                logging.info("No source_nodes attribute found in the chat response.")
            return contexts, relevance_scores

        context, relevance_scores = get_source_docs(llm_response)
        return context, llm_response.response