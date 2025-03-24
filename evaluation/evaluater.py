from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def evaluate_rag(retriever, num_questions: int = 5) -> Dict[str, Any]:
    """
    Evaluates a RAG system using predefined test questions and metrics.

    Args:
        retriever: The retriever component to evaluate
        num_questions: Number of test questions to generate

    Returns:
        Dict containing evaluation metrics
    """

    # Initialize LLM
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo-preview")

    # Create evaluation prompt
    eval_prompt = PromptTemplate.from_template("""
    Evaluate the following retrieval results for the question.

    Question: {question}
    Retrieved Context: {context}

    Rate on a scale of 1-5 (5 being best) for:
    1. Relevance: How relevant is the retrieved information to the question?
    2. Completeness: Does the context contain all necessary information?
    3. Conciseness: Is the retrieved context focused and free of irrelevant information?

    Provide ratings in JSON format:
    """)

    # Create evaluation chain
    eval_chain = (
            eval_prompt
            | llm
            | StrOutputParser()
    )

    # Generate test questions
    question_gen_prompt = PromptTemplate.from_template(
        "Generate {num_questions} diverse test questions about climate change:"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()

    questions = question_chain.invoke({"num_questions": num_questions}).split("\n")

    # Evaluate each question
    results = []
    for question in questions:
        # Get retrieval results
        context = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in context])

        # Evaluate results
        eval_result = eval_chain.invoke({
            "question": question,
            "context": context_text
        })
        results.append(eval_result)

    return {
        "questions": questions,
        "results": results,
        "average_scores": calculate_average_scores(results)
    }