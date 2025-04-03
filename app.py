import streamlit as st
from rag import get_rag_response

st.title("RAG Demo")
st.write("Ask me anything boo.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_rag_response(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})