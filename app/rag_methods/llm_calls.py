from llm_setup.llm_prompts import get_prompt
from llm_setup.setup_llm import prompt_llm
import ast
import re


def extract_metadata(client, query) -> dict:
    prompt = get_prompt('metadata_extraction', query=query)
    result = prompt_llm(client, prompt)
    result = ast.literal_eval(result)
    return result


def generate_hypothetical_document(llm_client, query):
    """
    Generates a hypothetical document (a detailed answer/summary) for the given query using an LLM.
    The prompt includes an example wine document solely as a template to illustrate the desired structure.
    The model is instructed not to copy the example verbatim but to produce new, original content.

    Args:
        query (str): The user's query.
        llm_client: An instance of an LLM client (e.g., OpenAI client).

    Returns:
        str: The generated hypothetical document.
    """

    prompt = get_prompt('generate_hypo', query=query)
    hypo_doc = prompt_llm(llm_client, prompt)
    return hypo_doc


def generate_queries_llm(client, original_query, num_queries=3):
    """
    Uses the LLM to generate multiple query variations from the original wine query.

    Args:
      client: The LLM client (e.g., from set_up_llm).
      original_query (str): The user's original wine recommendation query.
      num_queries (int): The number of query variations to generate.

    Returns:
      List[str]: A list of query variations.
    """
    prompt = get_prompt('generate_fusion_queries', original_query=original_query, num_queries=num_queries)
    response = prompt_llm(client, prompt)
    response = ast.literal_eval(response)
    return response


def rewrite_query_smart(client, original_query: str, context: list[str]) -> str:
    prompt = get_prompt('rewrite_query', original_query=original_query, context=chr(10).join(context))
    response = prompt_llm(client, prompt)
    return response


def get_recommendation(client, retrieval_context, query):
    final_prompt = get_prompt("final_recommendation", retrieval_context=retrieval_context, query=query)
    recommendation = prompt_llm(client, final_prompt)
    return recommendation
