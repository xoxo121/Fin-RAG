import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class HyDEGenerator:
    def __init__(self, model_name: str = "llama3-8b-8192"):
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.7,
            max_tokens=512,
            timeout=30,
            max_retries=3,
            stop_sequences=[]
        )
        self.model_name = model_name

    def generate_hypothetical_document(self, query: str, context: Optional[str] = None) -> str:
        system_prompt = """
        You are a knowledgeable expert in the field of finance.
        Given a question, generate a well-structured, informative paragraph that answers the question.
        Ensure accuracy and preserve any dates or numbers mentioned. The paragraph should be detailed and coherent.
        """

        if context:
            system_prompt += f"\n\nAdditional Context: {context}"

        messages = [
            ("system", system_prompt),
            ("human", f"Question: {query}\n\nParagraph:")
        ]

        try:
            response = self.llm.invoke(messages)
            if isinstance(response.content, list):
                content = [item if isinstance(item, str) else str(item) for item in response.content]
                generated_paragraph = ''.join(content).strip()
            else:
                generated_paragraph = str(response.content).strip()
            
            # Append the actual query to the generated paragraph
            return f"Original Query: {query}\n\n{generated_paragraph}"
        except Exception:
            return "I apologize, but I couldn't generate a hypothetical document at this time."