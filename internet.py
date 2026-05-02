import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from openai import OpenAI


load_dotenv(dotenv_path=".env")
api_key = os.getenv("OPENAI_API_KEY")

st.title("Internet Search Chatbot")

if not api_key:
    st.error("`.env` 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()

client = OpenAI(api_key=api_key)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("인터넷에서 찾아볼 내용을 입력하세요...")

if user_input:
    user_message = HumanMessage(content=user_input)
    st.session_state.chat_history.append(user_message)

    with st.chat_message("user"):
        st.markdown(user_input)

    conversation_lines = []
    for message in st.session_state.chat_history:
        speaker = "assistant" if isinstance(message, AIMessage) else "user"
        conversation_lines.append(f"{speaker}: {message.content}")
    conversation_text = "\n".join(conversation_lines)

    with st.chat_message("assistant"):
        with st.spinner("웹 검색 중..."):
            response = client.responses.create(
                model="gpt-5.2",
                tools=[{"type": "web_search"}],
                input=(
                    "아래 대화 맥락을 참고하여 최신 사용자 질문에 답변하세요.\n\n"
                    f"{conversation_text}"
                ),
            )
            answer = response.output_text
            st.markdown(answer)

    st.session_state.chat_history.append(AIMessage(content=answer))
