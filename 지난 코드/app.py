# 통합된 전체 app.py 코드입니다. 트럼프 스타일 상담, 언어 출력 설정(영/한/모두), 프로필 UI 등 반영됨

import os
import json
import random
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 🔐 API 키 로딩
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 트럼프 특유 표현들
TRUMP_REACTIONS = {
    "confident": ["Listen,", "Look,", "Here's what you do,", "This is easy,", "No problem!"],
    "supportive": ["Hey, that's tough, but", "I get it,", "Been there,", "Same thing happened to me,"],
    "dismissive": ["That's nothing!", "Piece of cake!", "Are you kidding me?", "That's it?"]
}

TRUMP_STORY_STARTERS = [
    "You know what? Same exact thing happened to me",
    "I had the EXACT same problem",
    "This reminds me of when I",
    "Funny you mention that, because I",
    "Let me tell you what I did when",
    "I've been in your shoes"
]

TRUMP_CONFIDENCE_BOOSTERS = [
    "You're gonna be fine", "This is nothing you can't handle",
    "You got this", "Easy win", "Total victory coming up",
    "You're stronger than you think", "This is your moment"
]

# ✅ JSON → 벡터스토어로 로딩
@st.cache_resource
def load_vectorstore():
    with open("trump_quotes.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        quote = item.get("quote_kr (트럼프식 번역)", "")
        story = item.get("story_kr", "")
        topic = item.get("topic", "")
        context = item.get("context", "")
        full_text = f"{quote}\n{story}\n({topic}) {context}"
        docs.append(Document(page_content=full_text, metadata={"topic": topic, "story_kr": story, "quote": quote}))
   
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db, data

vectorstore, _ = load_vectorstore()

# 문제 심각도 판단 함수
def get_problem_severity(user_input):
    serious_keywords = ["죽고싶", "포기", "절망", "우울", "못하겠", "실패", "망했"]
    medium_keywords = ["힘들", "어려워", "걱정", "불안", "스트레스", "문제"]
    
    user_lower = user_input.lower()
    if any(word in user_lower for word in serious_keywords):
        return "serious"
    elif any(word in user_lower for word in medium_keywords):
        return "medium"
    else:
        return "light"

# 트럼프 응답 생성
def generate_trump_response(user_input):
    matched_docs = vectorstore.similarity_search(user_input, k=5)
    best_story, best_quote = "", ""
    
    for doc in matched_docs:
        story = doc.metadata.get("story_kr", "")
        if len(story) > 10:
            best_story = story
            best_quote = doc.metadata.get("quote", "")
            break
    
    severity = get_problem_severity(user_input)
    reaction = random.choice(TRUMP_REACTIONS[severity if severity in TRUMP_REACTIONS else "confident"])
    story_starter = random.choice(TRUMP_STORY_STARTERS)
    booster = random.choice(TRUMP_CONFIDENCE_BOOSTERS)
    
    system_prompt = f"""
    You are Donald Trump giving confident, straightforward advice. Be bold and direct.

    IMPORTANT STYLE RULES:
    - Start with: "{reaction}"
    - Then: "{story_starter}" and tell the story from reference
    - End with: "{booster}"
    - NO analysis or complicated explanations
    - Be conversational and confident
    - Share YOUR experience like you're bragging
    - Give simple, direct action steps
    - Use "I", "me", "my" a lot
    - Keep it under 150 words total

    Reference story to use: {best_story}
    Reference quote: {best_quote}
    """

    en_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.9,
        max_tokens=150
    ).choices[0].message.content.strip()

    kr_prompt = f"""
    이 트럼프 발언을 한국어로 번역해줘. 
    
    스타일 가이드:
    - 반말, 친근하고 자신만만하게
    - "나도", "내가", "나는" 많이 사용
    - "완전", "진짜", "정말", "엄청" 같은 강조어 사용
    - 자랑스럽고 당당한 톤
    - 간단명료하게, 복잡한 설명 금지
    - 끝에 "믿어봐", "할 수 있어", "문제없어" 같은 격려
    
    원문: {en_response}
    """
    kr_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": kr_prompt}],
        temperature=0.8
    ).choices[0].message.content.strip()

    return en_response, kr_response

# Streamlit 앱 시작
st.set_page_config(page_title="MYGA: Make You Great Again", layout="wide")

st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg" width="120" style="border-radius: 60px; border: 3px solid #FF4C4C;" />
        <h2 style='margin-top: 10px;'>🇺🇸 MAKE YOU GREAT AGAIN</h2>
        <p><em>"Sometimes by losing a battle, you find a new way to win the war." – Donald J. Trump</em></p>
    </div>
""", unsafe_allow_html=True)

# 스타일
st.markdown("""
    <style>
    .chat-message {
        max-width: 70%;
        padding: 12px 16px;
        border-radius: 15px;
        margin-bottom: 10px;
        display: inline-block;
        word-wrap: break-word;
        font-size: 16px;
        line-height: 1.5;
    }
    .user { background-color: #D0E2FF; text-align: right; float: right; clear: both; color: #001F54; border: 2px solid #4D90FE; }
    .assistant { background-color: #FFD6D6; text-align: left; float: left; clear: both; color: #8B0000; border: 2px solid #FF4C4C; }
    .profile-header { display: flex; align-items: center; margin-bottom: 6px; }
    .profile-pic { width: 24px; height: 24px; border-radius: 12px; margin-right: 8px; border: 1px solid #aaa; }
    .profile-name { font-weight: bold; color: #8B0000; font-size: 14px; }
    .en-text { margin-bottom: 4px; font-style: italic; color: #666; }
    .kr-text { color: #8B0000; font-size: 16px; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0

# 사이드바
with st.sidebar:
    st.markdown("### 💬 트럼프 응답 언어 설정")
    lang_option = st.radio("언어 선택", ("🇺🇸 영어", "🇰🇷 한국어", "🌐 모두 보기"))

    st.markdown("### 📊 대화 통계")
    st.write(f"총 대화 수: {st.session_state.chat_count}")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.chat_count = 0
        st.rerun()

# 기존 대화 렌더링
for msg in st.session_state.messages:
    role = msg["role"]
    st.markdown(f'<div class="chat-message {role}">{msg["content"]}</div>', unsafe_allow_html=True)

# 입력 받기
user_input = st.chat_input("고민을 트럼프에게 털어놓아보세요 🇺🇸")

if user_input:
    st.session_state.chat_count += 1
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="chat-message user">{user_input}</div>', unsafe_allow_html=True)

    if len(st.session_state.messages) <= 2 and user_input.strip().lower() in ["안녕", "안녕하세요", "hi", "hello", "ㅎㅇ", "하이", "트럼프"]:
        en_response = "Hey there! Donald Trump here. You came to the right guy - I solve problems like nobody's business. What's eating you?"
        kr_response = "안녕! 나 도널드 트럼프야. 제대로 된 사람한테 왔네. 나처럼 문제 해결 잘하는 사람 없거든. 뭐가 문제야? 🔥"
    elif len(user_input.strip()) < 5:
        en_response = random.choice([
            "Come on, give me the real story!",
            "That's it? Talk to me properly!",
            "I need details! What's the actual problem?",
            "Don't waste my time - what happened?"
        ])
        kr_response = random.choice([
            "뭐야 이게? 제대로 말해봐!",
            "이런 걸로는 안 돼. 무슨 일인지 똑바로 말해!",
            "시간 낭비하지 말고, 정확히 뭔 일이야?",
            "자세히 말해야 도와주지!"
        ])
    else:
        with st.spinner("트럼프가 자신의 경험을 떠올리고 있습니다... 💭"):
            en_response, kr_response = generate_trump_response(user_input)

    # 언어 옵션에 따른 응답 구성
    if lang_option == "🇺🇸 영어":
        final_reply = f"""
        <div class="profile-header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg" class="profile-pic" />
            <span class="profile-name">Donald J. Trump</span>
        </div>
        <div class='en-text'>{en_response}</div>
        """
    elif lang_option == "🇰🇷 한국어":
        final_reply = f"""
        <div class="profile-header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg" class="profile-pic" />
            <span class="profile-name">도널드 J. 트럼프</span>
        </div>
        <div class='kr-text'>{kr_response}</div>
        """
    else:
        final_reply = f"""
        <div class="profile-header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg" class="profile-pic" />
            <span class="profile-name">도널드 J. 트럼프</span>
        </div>
        <div class='en-text'>{en_response}</div>
        <div class='kr-text'>{kr_response}</div>
        """

    st.session_state.messages.append({"role": "assistant", "content": final_reply})
    st.markdown(f'<div class="chat-message assistant">{final_reply}</div>', unsafe_allow_html=True)

