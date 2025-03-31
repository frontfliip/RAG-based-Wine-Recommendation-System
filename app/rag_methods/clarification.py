from llm_setup.llm_prompts import get_prompt
from llm_setup.setup_llm import prompt_llm
import re

def filter_answers(client, answers, questions):
    """
    Filters a list of answers using the associated questions,
    keeping only those that are deemed informative.
    """
    def is_informative_llm(client, answer, question):
        prompt = get_prompt("is_answer_informative", answer=answer, question=question)
        response = prompt_llm(client, prompt)
        return response.strip().lower() == "yes"

    filtered = []
    for answer, question in zip(answers, questions):
        if is_informative_llm(client, answer, question):
            filtered.append(answer.strip())

    return filtered

def generate_clarifying_questions(client, query, num_questions=3):
    prompt = get_prompt("generate_clarifying_questions", initial_query=query, number_of_questions=num_questions)
    clarifying_questions = prompt_llm(client, prompt)
    clarifying_questions = re.findall(r'^\d+\.\s*(.*)', clarifying_questions, re.MULTILINE)
    return clarifying_questions