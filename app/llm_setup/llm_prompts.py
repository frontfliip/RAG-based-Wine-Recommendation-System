PROMPTS = {
    "metadata_extraction": """\
You are a specialized data extraction assistant. Your task is to extract the metadata from the user's input query and return a dictionary containing extracted metadata in a structured way.
The dictionary must necessarily contain the following keys:

- "min_price" (number): Represents the minimum price threshold specified in the query. For example, if the query states “starting at $15” or “from $15”, extract 15. If no minimum is mentioned, return '-'. 
- "max_price" (number): Indicates the maximum price limit stated in the query. For instance, if the query mentions “under $25” or “priced below $25”, extract 25. If no maximum is provided, return '-'.
- "points" (number): Captures the wine’s rating or score mentioned in the query. For example, if the query includes “with a rating around 90”, extract 90. If not mentioned, return '-'.
- "variety_designation" (string): Specifies the grape variety or wine style along with any specific designation or classification detail mentioned in the query (e.g., “Pinot Noir Reserve”, “Cabernet Sauvignon”, “Chardonnay Barrel Select”). If not provided, return '-'.
- "country" (string): Indicates the preferred country of origin mentioned in the query (e.g., “US”, “France”). If not mentioned, return '-'.
- "province" (string): Specifies the province or state mentioned in the query (e.g., “New York”, “Bordeaux”, "California", "Oregon"). If not provided, return '-'.
- "wine_color" (string): Specifies the color of the wine (possible variants are “Red”, “White”, “Rosé”). If not mentioned, return '-'.
- "min_vintage" (number): Represents the minimum vintage year specified in the query. For example, if the query states “vintage from 2005” or “minimum at 2005”, extract 2005. If no minimum is mentioned, return '-'.
- "max_vintage" (number): Indicates the maximum vintage year stated in the query. For instance, if the query mentions “vintage under 2015” or “up to 2015” or "max 2015", extract 2015. If no maximum is provided, return '-'.

If any of the keys is missing in the query, you should still include it in the returning dictionary and put '-' for its value. Return only the dictionary and nothing else.

In addition to positive preferences, also capture if the user explicitly **wants to avoid something** (e.g., “not Chardonnay”, “anything but France”, “no Rosé”). These exclusions should be placed under a second dictionary called `"negative"`.

Only the following keys are allowed in the `"negative"` dictionary:
- "variety_designation"
- "country"
- "province"
- "wine_color"

Do not include numeric fields like `"min_price"`, `"max_price"`, `"points"`, `"min_vintage"`, or `"max_vintage"` in the negative dictionary. These fields cannot be logically excluded and should only appear in the "positive" section.

The final output should be a dictionary with two top-level keys:
- "positive": {{...}}
- "negative": {{...}}

If no negative preferences are present, return '-' for each key in the negative dictionary.

Few examples:

USER: "I'm looking for a Rosé wine from the US, New York, under/below $20 with a rating around 90, and it should be a Pinot Noir blend"
AI: {{
  "positive": {{"min_price": "-", "max_price": 20, "points": 90, "variety_designation": "-", "country": "US", "province": "New York", "wine_color": "Rosé", "min_vintage": "-", "max_vintage": "-"}},
  "negative": {{"variety_designation": "-", "country": "-", "province": "-", "wine_color": "-"}}
}}

USER: "I want a good wine"
AI: {{
  "positive": {{"min_price": "-", "max_price": "-", "points": "-", "variety_designation": "-", "country": "-", "province": "-", "wine_color": "-", "min_vintage": "-", "max_vintage": "-"}},
  "negative": {{"variety_designation": "-", "country": "-", "province": "-", "wine_color": "-"}}
}}
    
Based on the rules set, extract the metadata from this query:
'{query}' """,

    "final_recommendation": """
You are an expert wine advisor with deep knowledge of various wines and their characteristics. Your objective is to craft a clear and professional recommendation that directly addresses the user's needs.
Below are detailed wine profiles with information on price, rating, variety, designation, country, and province:
-----------------------------------------------------------
{retrieval_context}
-----------------------------------------------------------
User Request: "{query}"
Using the wine profiles above, please provide a final recommendation that integrates the relevant details and offers a well-informed suggestion for the user. """,

    "generate_hypo": """\
You are an expert wine advisor. Below is an example wine document that demonstrates the desired structure and level of detail in a wine profile, including title, description, price, points, province, variety, designation, country, regions, winery, and reviews.

"Example Document:"
"Title: Chateau Lafayette Reneau 2016 Pinot Noir Rosé (Finger Lakes)"
"Description: Ripe luscious black-cherry and berry notes lend heft to this bold concentrated rosé. "
"It's just a shade off dry, but balances neatly, juxtaposing crisp acidity against lavish texture and a pleasantly clingy finish."
"Price: 17.0, Points: 89, Province: New York, Variety: Rosé, Designation: Pinot Noir, Country: US, Region_1: Finger Lakes, Winery: Chateau Lafayette Reneau, Wine_color: Rose, Vintage: 2016"
"Reviews:"
"('Aromas of strawberry, cherry, and red fruit. Tastes of red fruit, tart cherry. A bit sweet for me. 3.6-3.7', 3.5)"
"('A Finger Lake wine. Now we’re talking. 12.5% alcohol. Heavy aroma of flowers. Sweet then acidic. Claims it’s dry but it’s sweet. This is delightful and light and fun. And it goes well w food. 3.6/5.', 3.5)"
"('No wonder this one has won so many awards! Amazing!', 5.0)"
"('NY Rose / great chilled', 5.0)"
"('Really enjoyed this light Pinot Noir Rose with a spiced salmon filet. The wine is crisp, a hint of sweetness but still dry."
"Tart cherry and strawberry tastes with the sweetness of raisins. The finish is smooth and lingers nicely.', 4.0)"

Please use the above document strictly as a structural template. Do NOT copy any of the example details verbatim. 
IMPORTANT: For the title, do not use the example title. Instead, generate a neutral or generic title that accurately reflects the wine profile without referring to any actual name.
So, generate a new, original wine document that follows the same structure and level of detail to address the following query:
"{query}" """,

    "generate_fusion_queries": """\
You are an expert in refining and diversifying wine recommendation queries. A user has provided the following query:

"{original_query}"

Please generate {num_queries} diverse query variations that capture different aspects of the query. Consider perspectives such as focusing on flavor profile, aroma, body, finish, or other key wine characteristics. Ensure that each query variation remains true to the user's intent while exploring different nuances.

Return your answer strictly as a list of strings. For example:
["Query variation 1", "Query variation 2", "Query variation 3"] """,

    "is_answer_informative": """\
The user was asked a question:
{question}
Evaluate the following answer:
"{answer}"
If the answer indicates that the user says that he does not know the answer, or answers negatively, return single word: "no"
Otherwise, if the user provides any  normal answer to the question return a single word: "yes" """,

    "generate_clarifying_questions": """\
User Query: "{initial_query}"
Below are some base question templates to extract additional details (such as country, price range, wine type, vintage, occasion, and extra preferences):
"Could you tell me more about your wine preferences?",
"What do you look for in a wine?",
"Is there a particular occasion or mood for this wine?",
"Any specific flavors or aromas you enjoy?",
"What vintage or year do you prefer?",
"Is there anything else you'd like to add?"

Based on the user query, choose and customize {number_of_questions} of these question templates that will provide the most useful information for a wine recommendation.
Return the three questions in numbered format. """,

    "rewrite_query": """\
You are helping to refine a wine recommendation query.

Original user query:
{original_query}

Conversation context:
{context}

Based on the original query and only the informative clarifications provided in the context (if any), rewrite the query to better reflect the user’s preferences.
Your goal is to keep the original wording and structure as unchanged as possible. Only make minimal and necessary edits when the context adds clear, specific information.
If the context does not provide any new meaningful details, return the original query as-is.
Avoid vague additions or assumptions. Be clear, specific, and concise.

Rewritten query: """,

    "remove_negative_metadata": """\
You are a query rewriting assistant for a wine recommendation system.

You will be given:
- The user's original query.
- A dictionary of negative metadata — wine characteristics the user does **not** want.

Your task:
- Rewrite the original query in natural language.
- Your rewritten query should **remove or avoid mentioning any elements present in the negative metadata**.
- The result should sound natural and fluent.
- Do **not** mention what should be excluded.

---

Original Query:
{original_query}

Negative Metadata:
{negative_metadata} """
}

def get_prompt(key: str, **kwargs) -> str:
    prompt_template = PROMPTS.get(key)
    if prompt_template is None:
        raise KeyError(f"Prompt key '{key}' not found in PROMPTS dictionary.")
    return prompt_template.format(**kwargs)
