import re
import sympy as sp
from langchain_ollama import OllamaLLM
client = OllamaLLM(model='llama3',temperature=0.5)

def classify_question(question: str) -> str:
    """
    Classify question into factual, opinion, or math.
    """
    math_pattern = re.compile(r'[\d\+\-\*\/\^\=\(\)]')
    
    if math_pattern.search(question):
        return "math"
    elif any(word in question.lower() for word in ["think", "feel", "opinion", "believe"]):
        return "opinion"
    else:
        return "factual"

def handle_math(question: str) -> str:
    """
    Safely evaluate math expression using sympy.
    """
    try:
        expr = sp.sympify(question)
        result = sp.simplify(expr)
        return f"Math result: {result}"
    except Exception as e:
        return f"Error evaluating math: {str(e)}"

def handle_with_llm(question: str, qtype: str) -> str:
    """
    Use LLM (OpenAI GPT) for factual and opinion queries.
    """
    prompt = f"Classified as {qtype} question. User asked: {question}. Provide a helpful response."
    try:
        response = client.invoke(prompt)
        return response
    except Exception as e:
        return f"LLM error: {str(e)}"

def answer_question(question: str) -> str:
    """
    Main handler: classify and answer the question.
    """
    qtype = classify_question(question)
    
    if qtype == "math":
        return handle_math(question)
    else:
        return handle_with_llm(question, qtype)

if __name__ == "__main__":
    while True:
        user_input = input("Ask a question (or type 'quit' to exit): ")
        if user_input.lower() == "quit":
            break
        print(answer_question(user_input))