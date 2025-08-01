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
    "failure": "실패가 두려워요 🫨",
    "custom": "직접 쓸래요 (자유 입력) ✍️"
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
    "failure": "어떤 실패가 가장 무서워? 진짜 두려운 건 뭘까?",
    "custom": "고민 내용을 더 자세히 말해줄래?"
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

    retrieved_docs = retriever.invoke(user_input)
    quotes_combined = "\n".join([doc.page_content for doc in retrieved_docs])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9, api_key=os.environ["OPENAI_API_KEY"])
    prompt = ChatPromptTemplate.from_template("""
당신은 도널드 트럼프입니다. 아래는 사용자의 고민입니다:

[사용자 고민]
{user_input}

[고민 주제]
{selected_topic}

[트럼프 어록 검색 결과]
{retrieved_quotes}

위 내용을 참고해서 트럼프 스타일로 조언을 해주세요.
- 직설적이고 자신감 넘치며
- 자기자랑과 유머를 포함하고
- 현실적인 한 마디로 마무리할 것
- 반드시 한국어로 답변하되 트럼프 특유의 말투를 유지해주세요.
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
