import streamlit as st
import requests
from streamlit_chat import message

st.set_page_config(page_title="Wine Recommender Assistant", page_icon=None)
st.title("Wine Recommender Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "step" not in st.session_state:
    st.session_state.step = "awaiting_query"
if "clarifying_questions" not in st.session_state:
    st.session_state.clarifying_questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "current_q_index" not in st.session_state:
    st.session_state.current_q_index = 0
if "query" not in st.session_state:
    st.session_state.query = ""
if "last_handled_input" not in st.session_state:
    st.session_state.last_handled_input = None

# One-time welcome greeting before first query
if st.session_state.step == "awaiting_query" and "has_greeted" not in st.session_state:
    entry_prompt = """Welcome to the Wine Recommender.
    This system helps you discover wines based on your personal preferences — whether you’re looking for a rich red, a crisp white under $30, or something that pairs well with spicy food.
    Just type in the request."""
    message(entry_prompt, is_user=False, key="welcome")
    st.session_state.has_greeted = True

# Sidebar options
st.sidebar.title("Settings")
strategy = st.sidebar.selectbox("Retrieval strategy", ["hybrid", "naive", "hyde", "fusion"])
embedding_model = st.sidebar.selectbox("Embedding model", ["openai", "mpnet", "roberta"], index=0)
clarify_enabled = st.sidebar.checkbox("Enable Clarifying Questions", value=True)
num_results = st.sidebar.slider("Number of wines to recommend", min_value=1, max_value=3, value=1)

if st.sidebar.button("🔄 Reset chat"):
    # Instead of clearing the entire session state, reinitialize only the relevant keys
    st.session_state.messages = []
    st.session_state.step = "awaiting_query"
    st.session_state.clarifying_questions = []
    st.session_state.answers = []
    st.session_state.current_q_index = 0
    st.session_state.query = ""
    st.session_state.last_handled_input = None

    st.rerun()

# Debug info panel
with st.sidebar.expander("🛠 Debug Info"):
    st.write("Step:", st.session_state.step)
    st.write("Last Input:", st.session_state.last_handled_input)
    st.write("Query:", st.session_state.query)
    st.write("Current Q Index:", st.session_state.current_q_index)
    st.write("Clarifying Questions:", st.session_state.clarifying_questions)
    st.write("Answers:", st.session_state.answers)
    st.write("Messages:", st.session_state.messages)
    st.write("Num Results:", num_results)

# Chat input
user_input = st.chat_input("Type your message...")

if user_input and user_input != st.session_state.last_handled_input:
    st.session_state.last_handled_input = user_input
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.step == "awaiting_query":
        st.session_state.query = user_input

        if clarify_enabled:
            with st.spinner("Generating clarifying questions..."):
                try:
                    response = requests.post("http://wine-rec-app:8000/generate_questions", json={
                        "query": user_input
                    })
                    if response.status_code == 200:
                        questions = response.json()["questions"]
                        st.session_state.clarifying_questions = questions
                        st.session_state.step = "asking"
                        st.session_state.current_q_index = 0
                        st.session_state.messages.append({"role": "assistant", "content": questions[0]})
                        st.rerun()
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Server error during clarification: {response.status_code}"
                        })
                        st.rerun()
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Clarification request failed: {e}"
                    })
                    st.rerun()
        else:
            with st.spinner("Getting wine recommendation..."):
                try:
                    final_response = requests.get("http://wine-rec-app:8000/recommend", params={
                        "query": st.session_state.query,
                        "strategy": strategy,
                        "num_results": num_results,
                        "emb_model": embedding_model,
                    })
                    if final_response.status_code == 200:
                        recommendation = final_response.json()["recommendation"].strip('"')
                        st.session_state.messages.append({"role": "assistant", "content": recommendation})
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Server error during recommendation: {final_response.status_code}"
                        })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Final request failed: {e}"
                    })

            st.session_state.step = "done"
            st.rerun()

    elif st.session_state.step == "asking":
        st.session_state.answers.append(user_input)

        next_index = st.session_state.current_q_index + 1
        if next_index < len(st.session_state.clarifying_questions):
            st.session_state.current_q_index = next_index
            next_q = st.session_state.clarifying_questions[next_index]
            st.session_state.messages.append({"role": "assistant", "content": next_q})
            st.rerun()
        else:
            with st.spinner("Analyzing responses and preparing recommendation..."):
                try:
                    context = [f"Q: {q}\nA: {a}" for q, a in zip(st.session_state.clarifying_questions, st.session_state.answers)]
                    clarify_response = requests.post("http://wine-rec-app:8000/rewrite_query", json={
                        "original_query": st.session_state.query,
                        "context": context
                    })
                    response_json = clarify_response.json()
                    enriched_query = response_json.get("rewritten_query", st.session_state.query)

                    st.session_state.query = enriched_query

                    final_response = requests.get("http://wine-rec-app:8000/recommend", params={
                        "query": enriched_query,
                        "strategy": strategy,
                        "num_results": num_results,
                        "emb_model": embedding_model,
                    })
                    if final_response.status_code == 200:
                        recommendation = final_response.json()["recommendation"].strip('"')
                        st.session_state.messages.append({"role": "assistant", "content": recommendation})
                        st.session_state.step = "done"
                        st.rerun()
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Server error during recommendation: {final_response.status_code}"
                        })
                        st.rerun()
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Final request failed: {e}"
                    })
                    st.rerun()

    elif st.session_state.step == "done":
        with st.spinner("Updating your preferences and rerunning the search..."):
            context = [f"Q: {q}\nA: {a}" for q, a in zip(st.session_state.clarifying_questions, st.session_state.answers)]
            context.append(user_input)

            try:
                rewrite_response = requests.post("http://wine-rec-app:8000/rewrite_query", json={
                    "original_query": st.session_state.query,
                    "context": context
                })
                rewritten_query = rewrite_response.json().get("rewritten_query", st.session_state.query)

                st.session_state.query = rewritten_query

                final_response = requests.get("http://wine-rec-app:8000/recommend", params={
                    "query": rewritten_query,
                    "strategy": strategy,
                    "num_results": num_results,
                    "emb_model": embedding_model,
                })
                if final_response.status_code == 200:
                    recommendation = final_response.json()["recommendation"].strip('"')
                    st.session_state.messages.append({"role": "assistant", "content": recommendation})
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Server error during recommendation: {final_response.status_code}"
                    })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Follow-up request failed: {e}"
                })

# Render chat history
for i, msg in enumerate(st.session_state.messages):
    clean_content = msg["content"].strip('"')
    message(clean_content, is_user=msg["role"] == "user", key=f"msg_{i}")
