# MYGA TWEETS - Streamlit 앱
import streamlit as st
import json
import os
import random
import html
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# 🔐 환경변수 로딩
load_dotenv()

# ------------------- 페이지 설정 -------------------
st.set_page_config(page_title="MYGA TWEETS", layout="centered")

# ------------------- 트위터 스타일 CSS -------------------
st.markdown("""
<style>
.stApp {
    background-color: #000000;
}

.main .block-container {
    padding-top: 0;
    padding-bottom: 0;
    max-width: 600px;
}

.trump-header {
    background-color: #000000;
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
}

.trump-banner {
    width: 100%;
    height: 200px;
    object-fit: cover;
    border-radius: 0;
}

.trump-profile-section {
    padding: 12px 16px 0 16px;
    position: relative;
}

.trump-profile-img {
    width: 134px;
    height: 134px;
    border: 4px solid #000000;
    border-radius: 50%;
    position: absolute;
    top: -67px;
    left: 16px;
    background-size: cover;
    background-position: center;
}

.follow-btn {
    position: absolute;
    top: 12px;
    right: 16px;
    background-color: #1d9bf0;
    color: white;
    border: none;
    border-radius: 20px;
    padding: 6px 16px;
    font-weight: bold;
    font-size: 15px;
    cursor: pointer;
}

.trump-profile-info {
    margin-top: 75px;
    padding-bottom: 12px;
}

.trump-name {
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
    line-height: 24px;
    display: flex;
    align-items: center;
    gap: 4px;
}

.verified-badge {
    width: 16px;
    height: 16px;
    background-color: #1d9bf0;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 10px;
    font-weight: bold;
}

.trump-handle {
    color: #71767b;
    font-size: 15px;
    margin: 0;
    line-height: 20px;
}

.trump-bio {
    color: #ffffff;
    font-size: 15px;
    line-height: 20px;
    margin: 12px 0;
}

.trump-location-info {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    color: #71767b;
    font-size: 15px;
    margin: 12px 0;
    align-items: center;
}

.trump-follow-info {
    display: flex;
    gap: 20px;
    margin: 12px 0;
    font-size: 15px;
}

.trump-follow-info span {
    color: #71767b;
}

.trump-follow-info strong {
    color: #ffffff;
    font-weight: 700;
}

.trump-tabs {
    display: flex;
    border-bottom: 1px solid #2f3336;
    margin-top: 16px;
}

.trump-tab {
    flex: 1;
    text-align: center;
    padding: 16px;
    color: #71767b;
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: color 0.2s;
}

.trump-tab.active {
    color: #ffffff;
    border-bottom-color: #1d9bf0;
    font-weight: 700;
}

.trump-tab:hover {
    background-color: #080808;
}

.timeline {
    background-color: #000000;
}

.user-tweet, .trump-tweet {
    background-color: #000000;
    border-bottom: 1px solid #2f3336;
    padding: 12px 16px;
    display: flex;
    gap: 12px;
}

.user-avatar, .trump-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    flex-shrink: 0;
}

.user-avatar {
    background-color: #1d9bf0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
}

.trump-avatar {
    object-fit: cover;
}

.tweet-content {
    flex: 1;
}

.tweet-header {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 2px;
}

.tweet-username {
    color: #ffffff;
    font-weight: 700;
    font-size: 15px;
}

.tweet-handle {
    color: #71767b;
    font-size: 15px;
}

.tweet-text {
    color: #ffffff;
    font-size: 15px;
    line-height: 20px;
    margin: 2px 0 12px 0;
}

.tweet-actions {
    display: flex;
    justify-content: space-between;
    max-width: 425px;
    margin-top: 12px;
}

.tweet-action {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #71767b;
    font-size: 13px;
    cursor: pointer;
}

.tweet-action:hover {
    color: #1d9bf0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='trump-header'>
    <img src='https://pbs.twimg.com/profile_banners/25073877/1604214583/1500x500' class='trump-banner'>
    <div class='trump-profile-section'>
        <div class='trump-profile-img' style='background-image: url("https://pbs.twimg.com/profile_images/874276197357596672/kUuht00m_400x400.jpg");'></div>
        <button class='follow-btn'>팔로우</button>
        <div class='trump-profile-info'>
            <div class='trump-name'>Donald J. Trump <span class='verified-badge'>✓</span></div>
            <div class='trump-handle'>@realDonaldTrump</div>
            <div class='trump-bio'>45th & 47th President of the United States of America🇺🇸</div>
            <div class='trump-location-info'>📍 Washington, DC · 🔗 Vote.DonaldJTrump.com · 📅 가입일: 2009년 3월</div>
            <div class='trump-follow-info'>
                <span><strong>53</strong> 팔로우 중</span>
                <span><strong>1억</strong> 팔로워</span>
            </div>
        </div>
    </div>
    <div class='trump-tabs'>
        <div class='trump-tab active'>게시물</div>
        <div class='trump-tab'>답글</div>
        <div class='trump-tab'>하이라이트</div>
        <div class='trump-tab'>미디어</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------- 벡터스토어 -------------------
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

# ------------------- 세션 초기화 -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------- 입력 -------------------
st.markdown('<div style="background-color: #000000; padding: 12px 16px;">', unsafe_allow_html=True)
user_input = st.text_input("", placeholder="@realDonaldTrump 요즘 너무 지쳐요...", key="user_input")
if st.button("트윗하기", type="primary") and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    retrieved_docs = retriever.vectorstore.similarity_search(query=user_input, k=3)
    quotes_combined = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = ChatPromptTemplate.from_template("""
너는 도널드 트럼프다. 사용자 고민에 트윗처럼 짧고 강하게, 트럼프식으로 조언해라.

[고민 내용]
{user_input}

[트럼프 어록 및 경험]
{retrieved_quotes}

조건:
- 문장은 짧게 끊어라 (트윗 스타일)
- 자기자랑 포함 (ex. "내가 해봤다니까")
- story_kr 활용한 경험담을 자연스럽게 녹여라
- 가벼운 욕설 허용 (빌어먹을, 헛소리 등)
- 마지막 문장은 "팩트"처럼 강하게 마무리
- 반드시 한국어로, 트럼프 말투로 작성
""")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0, api_key=os.environ["OPENAI_API_KEY"])
    chain = prompt | llm
    response = chain.invoke({"user_input": user_input, "retrieved_quotes": quotes_combined})
    st.session_state.chat_history.append({"role": "trump", "content": response.content})
st.markdown('</div>', unsafe_allow_html=True)

# ------------------- 출력 -------------------
st.markdown('<div class="timeline">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class='user-tweet'>
            <div class='user-avatar'>You</div>
            <div class='tweet-content'>
                <div class='tweet-header'>
                    <span class='tweet-username'>You</span>
                    <span class='tweet-handle'>@you</span>
                    <span class='tweet-handle'>·</span>
                    <span class='tweet-handle'>지금</span>
                </div>
                <div class='tweet-text'>{html.escape(msg['content'])}</div>
                <div class='tweet-actions'>
                    <div class='tweet-action'>💬 24</div>
                    <div class='tweet-action'>🔁 156</div>
                    <div class='tweet-action'>❤️ 1.2K</div>
                    <div class='tweet-action'>📊 89K</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='trump-tweet'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg' class='trump-avatar'>
            <div class='tweet-content'>
                <div class='tweet-header'>
                    <span class='tweet-username'>Donald J. Trump</span>
                    <span class='verified-badge'>✓</span>
                    <span class='tweet-handle'>@realDonaldTrump</span>
                    <span class='tweet-handle'>·</span>
                    <span class='tweet-handle'>지금</span>
                </div>
                <div class='tweet-text'>{html.escape(msg['content'])}</div>
                <div class='tweet-actions'>
                    <div class='tweet-action'>💬 11.2K</div>
                    <div class='tweet-action'>🔁 87.1K</div>
                    <div class='tweet-action'>❤️ 234K</div>
                    <div class='tweet-action'>📊 2.3M</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------- 초기화 -------------------
if st.button("🧹 타임라인 초기화"):
    st.session_state.chat_history = []
    st.rerun()
