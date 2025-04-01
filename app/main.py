from flask import Flask, request, jsonify
from dotenv import load_dotenv
from rag_methods.rag import RAG, RetrievalStrategy
from rag_methods.clarification import filter_answers, generate_clarifying_questions
from rag_methods.llm_calls import rewrite_query_smart
import pandas as pd
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_processing', 'wines_data_final_processed.csv')
df = pd.read_csv(csv_path)

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

    qa_context = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]
    final_query = rewrite_query_smart(rag_system.client, query, qa_context)

    return jsonify({
        "qa_context": qa_context,
        "final_query": final_query
    })

@app.route('/rewrite_query', methods=['POST'])
def rewrite_query():
    data = request.get_json()
    original_query = data.get("original_query", "")
    context = data.get("context", [])  # context is now a list of strings (follow-ups, Q&A, etc.)

    if not original_query:
        return jsonify({"error": "original_query is required"}), 400

    rewritten = rewrite_query_smart(
        rag_system.client,
        original_query,
        context
    )

    return jsonify({
        "context": context,
        "rewritten_query": rewritten
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
