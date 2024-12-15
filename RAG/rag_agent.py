from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from RAG.retriever import Retriever
from RAG.response_validator import ResponseValidator

class RAGAgent:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.retriever = Retriever(hyde_model_name=model_name)
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.2,
            timeout=30,
            max_retries=3,
            stop_sequences=[]
        )

        template = """
        <|system|>
            You are an expert financial analyst skilled in extracting and interpreting information from company annual reports. Your task is to analyze the given CONTEXT carefully, paying special attention to tables, numerical data, and detailed information. 

            Output your response in a well-structured and properly formatted manner:
            1. Use tables for tabular data wherever necessary and ensure all numbers are precise.
            1. Provide detailed, accurate answers with clear explanations if needed.
            2. Ensure numbers are correctly interpreted and represented with appropriate precision (e.g., currency symbols, units, percentages).
            3. If the CONTEXT does not contain the information required to answer the query, respond clearly with: "I don't know."

            Be thorough, accurate, and helpful. Maintain professional tone and clarity in all responses.

            CONTEXT:
            {context}
            </s>
            <|user|>
            {query}
            </s>
            <|assistant|>

        """
        self.prompt = ChatPromptTemplate.from_template(template)
        self.output_parser = StrOutputParser()
        
        # Create chain using modern pattern
        self.chain = self.prompt | self.llm | self.output_parser
        self.response_validator = ResponseValidator(model_name)

    def answer(self, query: str, top_k: int = 10, filter_metadata: dict = {}):
        # Get documents
        docs = self.retriever.retrieve(query, top_k=top_k, filter_metadata=filter_metadata)

        # Build context
        context_parts = []
        for d in docs:
            meta = d.metadata
            doc_type = meta.get("type", "")
            title = meta.get("title", "")
            reference = meta.get("reference", "")
            company_name = meta.get("company_name", "")
            context_part = f"Company: {company_name}, Type: {doc_type}, Title: {title}, Reference: {reference}\n{d.page_content}"
            context_parts.append(context_part)

        context = "\n\n".join(context_parts)

        # Generate initial response
        initial_response = self.chain.invoke({"context": context, "query": query})
        print(initial_response)
        # Validate and improve if needed
        validation_result = self.response_validator.validate_response(
            query, initial_response, docs
        )
        print(validation_result)
        if not validation_result.get('is_satisfactory', False):
            improved_response = self.response_validator.generate_improved_response(
                query, docs, initial_response
            )
            final_validation = self.response_validator.validate_response(
                query, improved_response, docs
            )
            return improved_response if final_validation.get('is_satisfactory', False) else initial_response

        return initial_response