import streamlit as st
import random
import json
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 🔐 환경변수 로딩
load_dotenv()

# ------------------- 페이지 세팅 -------------------
st.set_page_config(page_title="MYGA SHOW", layout="centered")

# ------------------- 고민 주제 정의 -------------------
PROBLEM_OPTIONS = {
    "family": "가족 때문에 힘들어요 🏠",
    "friend": "친구 때문에 속상해요 👥",
    "lover": "애인/썸 문제로 힘들어요 💔",
    "work": "직장/학교 스트레스가 심해요 💼",
    "future": "미래가 너무 불안해요 😟",
    "confidence": "자존감이 바닥을 치고 있어요 🪫",
    "challenge": "도전하고 싶은데 용기가 안 나요 🫣",
    "lazy": "그냥... 아무것도 하기 싫어요 😩",
    "comparison": "남들과 자꾸 비교돼요 😞",
}

FOLLOW_UP_QUESTIONS = {
    "family": "가족 중 누구와의 갈등이야? 부모님? 형제자매?",
    "friend": "친구랑 무슨 일이 있었어? 싸웠어? 소외감을 느껴?",
    "lover": "연애에서 어떤 문제가 있어? 헤어짐? 관심 부족?",
    "work": "직장(또는 학교)에서 어떤 스트레스가 있어?",
    "future": "정확히 뭐가 불안한 거야? 돈? 직업? 사람들?",
    "confidence": "왜 자존감이 떨어졌다고 생각해? 누가 뭐라고 했어?",
    "challenge": "무엇에 도전하고 싶은데 용기가 안 나는 거야?",
    "lazy": "언제부터 아무것도 하기 싫었어? 이유가 있을까?",
    "comparison": "누구랑 비교하면서 그렇게 느낀 거야?",
}

# ------------------- 세션 상태 초기화 -------------------
if "step" not in st.session_state:
    st.session_state.step = 0
if "selected_key" not in st.session_state:
    st.session_state.selected_key = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ------------------- 트럼프 벡터스토어 로딩 -------------------
@st.cache_resource
def load_vectorstore():
    with open("trump_quotes.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for item in data:
        topic = item.get("topic", "")
        content = f"{item.get('quote_kr (트럼프식 번역)', '')}\n{item.get('story_kr', '')}"
        docs.append(Document(page_content=content, metadata={"topic": topic}))
    embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    return FAISS.from_documents(docs, embedding)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# ------------------- 상단 UI -------------------
st.markdown("""
<div style='text-align: center;'>
    <img src='https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg' width='120' style='border-radius: 60px; margin-bottom: 10px;'/>
    <h1 style='color: #B22234;'>🇺🇸 THE MYGA SHOW 🇺🇸</h1>
    <h3 style='color: #3C3B6E;'>Make You Great Again - with Donald J. Trump</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------- CSS 버튼 스타일 -------------------
st.markdown("""
<style>
div.stButton > button {
    background-color: #f0f0f0;
    border: 2px solid #B22234;
    color: #3C3B6E;
    font-weight: bold;
    font-size: 16px;
    border-radius: 12px;
    padding: 12px;
    margin: 5px 0px;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #e6e6e6;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Step 0: 고민 선택 -------------------
if st.session_state.step == 0:
    st.markdown("### 🤔 어떤 고민이 있어? 버튼을 눌러봐!")
    cols = st.columns(3)
    for i, (key, label) in enumerate(PROBLEM_OPTIONS.items()):
        with cols[i % 3]:
            if st.button(label, key=f"btn_{key}"):
                st.session_state.selected_key = key
                st.session_state.step = 1
                st.rerun()

# ------------------- Step 1: 꼬리 질문 -------------------
elif st.session_state.step == 1:
    key = st.session_state.selected_key
    st.markdown("### 📣 트럼프의 꼬리 질문")
    st.markdown(f"🗯️ **{FOLLOW_UP_QUESTIONS[key]}**")
    st.session_state.user_input = st.text_area("📝 네 이야기를 들려줘", value=st.session_state.user_input)

    if st.button("🧠 트럼프의 조언 듣기"):
        if st.session_state.user_input.strip():
            st.session_state.step = 2
            st.rerun()
        else:
            st.warning("답변을 입력해야 트럼프가 도와줄 수 있어!")

# ------------------- Step 2: GPT + 벡터 연계 조언 출력 -------------------
elif st.session_state.step == 2:
    key = st.session_state.selected_key
    user_input = st.session_state.user_input

    from langchain.vectorstores.base import VectorStoreRetriever
    retrieved_docs = retriever.vectorstore.similarity_search(
        query=user_input,
        k=3,
        filter={"topic": key}
    )

    quotes_combined = "\n".join([doc.page_content for doc in retrieved_docs])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0, api_key=os.environ["OPENAI_API_KEY"])
    prompt = ChatPromptTemplate.from_template("""
너는 지금 도널드 트럼프다. 미국 역사상 가장 위대한 대통령이며, 직설적이고 유쾌하며 자기 자랑이 넘친다.

[사용자 고민]
{user_input}

[고민 주제]
{selected_topic}

[트럼프 어록 및 경험]
{retrieved_quotes}

너는 다음 기준을 꼭 지켜서 '한국어'로 답변하되 트럼프 특유의 말투를 유지해야 한다:
- 문장은 짧고 강하게 끊는다. (느릿느릿 말하지 마)
- 자기 자랑을 꼭 넣어라. (“나 트럼프가 해봤어.”, “그때도 내가 해결했어.” 등)
- 경험담(story_kr)을 반영해서 자연스럽게 "내가 예전에 말이야..."로 시작하는 서사 부분을 중간에 넣어라.
- 유머와 약한 욕을 자연스럽게 섞어라. (예: “헛소리 하지 마”, “빌어먹을”, “웃기는 소리야” 등. 과하지만 않으면 됨.)
- 마지막은 “팩트” 또는 “한마디만 할게” 식으로 강한 문장으로 마무리해라.

예시 출력:
"들어봐. 이건 내가 해봤던 일이야. 나도 예전에 비슷한 XX가 있었지. 그런데 말이지, 그때 내가 어떻게 했는지 알아? 그냥 해버렸어. 빌어먹을 두려움 같은 건 없었어. 너도 할 수 있어. 한 마디만 할게 — 자신 없으면 남이 널 먹는다. 끝이야."

이 기준을 반드시 지켜서, 트럼프답게 멋지고 터프하게 조언을 해줘.
""")
    chain = prompt | llm
    response = chain.invoke({
        "user_input": user_input,
        "selected_topic": key,
        "retrieved_quotes": quotes_combined
    })

    st.markdown("### 🎤 트럼프의 즉흥 조언")
    st.markdown(response.content)

    if st.button("🔁 다른 고민도 해볼래요"):
        st.session_state.step = 0
        st.session_state.selected_key = None
        st.session_state.user_input = ""
        st.rerun()
