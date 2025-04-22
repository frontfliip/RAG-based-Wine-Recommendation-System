from llm_setup.llm_prompts import get_prompt
from llm_setup.setup_llm import prompt_llm
import re


def generate_clarifying_questions(client, query, num_questions=3):
    prompt = get_prompt("generate_clarifying_questions", initial_query=query, number_of_questions=num_questions)
    clarifying_questions = prompt_llm(client, prompt)
    clarifying_questions = re.findall(r'^\d+\.\s*(.*)', clarifying_questions, re.MULTILINE)
    return clarifying_questions