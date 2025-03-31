from flask import Flask, request, jsonify
from dotenv import load_dotenv
from rag_methods.rag import RAG, RetrievalStrategy
from rag_methods.clarification import filter_answers, generate_clarifying_questions
import pandas as pd
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

df = pd.read_csv("../data_processing/wines_data_final_processed.csv")
rag_system = RAG(df=df, retrieval_strategy=RetrievalStrategy.HYBRID, k=3)

@app.route('/recommend', methods=['GET'])
def recommend():
    query = request.args.get('query')
    strategy = request.args.get('strategy', 'hyde').lower()

    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    try:
        rag_system.set_retrieval_strategy(strategy)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    recommendation = rag_system.recommend(query)

    return jsonify({
        'query': query,
        'strategy': strategy,
        'recommendation': recommendation
    })

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query field is required."}), 400

    questions = generate_clarifying_questions(rag_system.client, query)

    return jsonify({
        "questions": questions
    })

@app.route('/finalize_query', methods=['POST'])
def finalize_query():
    data = request.get_json()
    query = data.get("query")
    answers = data.get("answers", [])
    questions = data.get("questions", [])

    if not query:
        return jsonify({"error": "Query field is required."}), 400

    informative_answers = filter_answers(rag_system.client, answers, questions)
    print("Something", flush=True)
    final_query = query + "\n" + "\n".join(informative_answers) if informative_answers else query

    return jsonify({
        "filtered_answers": informative_answers,
        "final_query": final_query
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
