from .aimon_evaluator import AIMonEvaluator

class GuidelineEvaluator(AIMonEvaluator):
    
    def __init__(self, aimon_client, publish: bool = False) -> None:
        super().__init__(aimon_client, publish)

    def create_payload(self, context, user_query, user_instructions, generated_text) -> dict:
        
        aimon_payload = super().create_payload(context, user_query, user_instructions, generated_text)
        
        aimon_payload['config'] = {'instruction_adherence': {'detector_name': 'default'}}
        
        return aimon_payload
