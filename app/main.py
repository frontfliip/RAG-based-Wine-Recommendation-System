from flask import Flask, request, jsonify
from dotenv import load_dotenv
from rag_methods.rag import RAG, RetrievalStrategy
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

print("Files in FAISS directory:", os.listdir("vectorstore/faiss_index"))

app = Flask(__name__)

rag_system = RAG(retrieval_strategy=RetrievalStrategy.HYBRID)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)