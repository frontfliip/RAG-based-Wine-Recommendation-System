from llm_setup.llm_prompts import get_prompt
from llm_setup.setup_llm import prompt_llm
import ast


def extract_metadata(client, query):
    prompt = get_prompt('metadata_extraction', query=query)
    result = prompt_llm(client, prompt)
    result = ast.literal_eval(result)
    return result


def generate_hypothetical_document(llm_client, query):
    prompt = get_prompt('generate_hypo', query=query)
    hypo_doc = prompt_llm(llm_client, prompt)
    return hypo_doc


def generate_queries_llm(client, original_query, num_queries=3):
    prompt = get_prompt('generate_fusion_queries', original_query=original_query, num_queries=num_queries)
    response = prompt_llm(client, prompt)
    response = ast.literal_eval(response)
    return response


def rewrite_query_smart(client, original_query, context):
    prompt = get_prompt('rewrite_query', original_query=original_query, context=chr(10).join(context))
    response = prompt_llm(client, prompt)
    return response


def rewrite_query_remove_negative_metadata(client, original_query, negative_metadata):
    def format_metadata(md):
        return "\n".join(
            f"- {key}: {value}" for key, value in md.items() if value != "-"
        )

    prompt = get_prompt('remove_negative_metadata', original_query=original_query,
                        negative_metadata=format_metadata(negative_metadata))

    return prompt_llm(client, prompt)


def classify_query_intent(client, query):
    prompt = get_prompt("classify_query_intent", query=query)
    response = prompt_llm(client, prompt)
    response = ast.literal_eval(response)
    return response


def get_recommendation(client, retrieval_context, query, reference_doc=None, reference_wine_present=False,
                       num_results=1):
    plural_suffix = "s" if num_results > 1 else ""

    if reference_wine_present:
        final_prompt = get_prompt(
            "final_recommendation_with_reference",
            retrieval_context=retrieval_context,
            query=query,
            reference_wine=reference_doc,
            num_results=num_results,
            plural_suffix=plural_suffix
        )
    else:
        final_prompt = get_prompt(
            "final_recommendation",
            retrieval_context=retrieval_context,
            query=query,
            num_results=num_results,
            plural_suffix=plural_suffix
        )

    recommendation = prompt_llm(client, final_prompt)
    return recommendation
