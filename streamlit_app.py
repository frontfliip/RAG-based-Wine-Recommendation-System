import streamlit as st
import requests
from streamlit_chat import message

st.set_page_config(page_title="Wine Recommender Assistant", page_icon="🍷")
st.title("🍷 Wine Recommender Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar options
st.sidebar.title("Settings")
strategy = st.sidebar.selectbox("Retrieval strategy", ["hybrid", "naive", "hyde", "fusion"])

# Chat interface
for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=f"msg_{i}")

query = st.chat_input("What kind of wine are you looking for?")
if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    message(query, is_user=True)

    with st.spinner("Getting wine recommendation..."):
        try:
            response = requests.get(
                "http://localhost:8000/recommend",
                params={"query": query, "strategy": strategy}
            )
            if response.status_code == 200:
                recommendation = response.json()["recommendation"]
                st.session_state.messages.append({"role": "assistant", "content": recommendation})
                message(recommendation)
            else:
                error_msg = f"Server error {response.status_code}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                message(error_msg)
        except Exception as e:
            error_msg = f"Request failed: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            message(error_msg)
