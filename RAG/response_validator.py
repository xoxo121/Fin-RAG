from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import json

class ResponseValidator:
    def __init__(self, model_name="llama3-70b-8192", max_retries=3):
        self.judge_llm = ChatGroq(
            model=model_name,
            temperature=0.1,
            max_tokens=512,
            stop_sequences=[]
        )
        self.generation_llm = ChatGroq(
            model=model_name,
            temperature=0.2,
            max_tokens=512,
            stop_sequences=[]
        )
        self.max_retries = max_retries

    def validate_response(
        self, 
        query: str, 
        original_response: str, 
        context_docs: List[Document]
    ) -> Dict[str, Any]:
        validation_prompt = f"""
        Act as a critical judge evaluating a response to a query.

        Evaluation Criteria:
        1. Relevance: Does the response directly address the query?
        2. Accuracy: Is the information factually correct based on the given context?
        3. Completeness: Does the response cover all key aspects of the query?
        4. Coherence: Is the response well-structured and logically sound?

        Query: {query}
        
        Context Documents:
        {' '.join([doc.page_content for doc in context_docs])}

        Response to Evaluate:
        {original_response}

        Provide a detailed evaluation with:
        - overall_score (0-100)
        - is_satisfactory (bool)
        - strengths (list)
        - weaknesses (list)
        - confidence (0.0-1.0)

        Output Format (JSON):
        {{
            "overall_score": int,
            "is_satisfactory": bool,
            "strengths": [str],
            "weaknesses": [str],
            "confidence": float
        }}
        """

        try:
            validation_response = self.judge_llm.invoke(validation_prompt)
            validation_result = validation_response.content
            if isinstance(validation_result, list):
                validation_result = json.dumps(validation_result)
            return self._parse_validation_result(validation_result)
        except Exception:
            return {
                "overall_score": 50,
                "is_satisfactory": False,
                "strengths": [],
                "weaknesses": ["Validation failed"],
                "confidence": 0.5
            }

    def _parse_validation_result(self, validation_text: str) -> Dict[str, Any]:
        try:
            return json.loads(validation_text)
        except:
            return {
                "overall_score": 50,
                "is_satisfactory": False,
                "strengths": [],
                "weaknesses": ["Unable to parse validation result"],
                "confidence": 0.5
            }

    def generate_improved_response(
        self, 
        query: str, 
        context_docs: List[Document], 
        previous_response: str
    ) -> str:
        improvement_prompt = f"""
        You previously generated a response that was deemed partially unsatisfactory.
        
        Original Query: {query}
        
        Context Documents:
        {' '.join([doc.page_content for doc in context_docs])}

        Previous Response:
        {previous_response}

        Instructions:
        1. Carefully review the previous response
        2. Identify and address any gaps or weaknesses
        3. Provide a more comprehensive, accurate, and relevant response
        4. Directly address the query
        5. Use information from the context documents if available
        6. Output in a well-structured manner, and use JSON for any tabular data.

        Improved Response:
        """

        try:
            improved_response = self.generation_llm.invoke(improvement_prompt)
            return str(improved_response.content).strip()
        except Exception:
            return previous_response
