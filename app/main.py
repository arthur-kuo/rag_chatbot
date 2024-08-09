import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dependencies.chain import get_response
from dotenv import load_dotenv

load_dotenv()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title='Streaming Bot')
st.title('Streaming Bot')


for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message('Human'):
            st.markdown(message.content)
    else:
        with st.chat_message('AI'):
            st.markdown(message.content)

user_query = st.chat_input('Your message')
if user_query is not None and user_query != '':
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message('Human'):
        st.markdown(user_query)
    with st.chat_message('AI'):
        ai_response = st.write_stream(get_response(user_query, '123'))
        ai_response_str = ''.join(ai_response)
        st.session_state.chat_history.append(AIMessage(ai_response_str))
