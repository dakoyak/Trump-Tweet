# MYGA TWEETS - Streamlit 앱 (완전한 RAG 시스템)
import streamlit as st
import random
import os
import html
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔐 환경변수 로딩
load_dotenv()

# ------------------- 페이지 설정 -------------------
st.set_page_config(page_title="MYGA TWEETS", layout="centered")

# ------------------- 트위터 스타일 CSS (동일) -------------------
st.markdown("""
<style>
.stApp {
    background-color: #000000;
}

.main .block-container {
    padding-top: 0;
    padding-bottom: 100px; /* 입력창 공간 확보 */
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
    white-space: pre-wrap;
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

/* 하단 고정 입력창 스타일 */
.fixed-input-container {
    position: fixed;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    max-width: 600px;
    background-color: #000000;
    border-top: 1px solid #2f3336;
    padding: 12px 16px;
    z-index: 1000;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.5);
}

/* Streamlit 컴포넌트 스타일 수정 */
.fixed-input-container .stTextInput > div > div > input {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #2f3336 !important;
    border-radius: 20px !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
}

.fixed-input-container .stTextInput > div > div > input::placeholder {
    color: #71767b !important;
}

.fixed-input-container .stButton > button {
    background-color: #1d9bf0 !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 8px 24px !important;
    font-weight: bold !important;
    font-size: 15px !important;
    margin-top: 8px !important;
}

.fixed-input-container .stButton > button:hover {
    background-color: #1a8cd8 !important;
}

/* 로딩 스피너 */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #2f3336;
    border-radius: 50%;
    border-top-color: #1d9bf0;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# ------------------- 헤더 및 프로필 섹션 -------------------
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

# ------------------- RAG 시스템 구축 -------------------
@st.cache_resource
def setup_trump_rag_system():
    faiss_index_path = "trump_faiss_index"
    
    try:
        # 1. 기존 벡터DB가 있으면 로드
        if os.path.exists(faiss_index_path):
            embeddings = OpenAIEmbeddings(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                model="text-embedding-3-small"
            )
            vectorstore = FAISS.load_local(
                faiss_index_path, 
                embeddings,
                allow_dangerous_deserialization=True  # 로컬 파일이므로 안전
            )
            return vectorstore
        
        # 텍스트 파일 읽기
        with open("trump_all.txt", "r", encoding="utf-8") as f:
            trump_data = f.read()
        
        # 카테고리별로 분할
        categories = trump_data.split("### [CATEGORY] ")
        
        # Document 객체 생성
        documents = []
        for i, category_chunk in enumerate(categories[1:]):
            lines = category_chunk.strip().split('\n')
            if len(lines) < 2:
                continue
                
            category_name = lines[0].strip()
            category_content = '\n'.join(lines[1:]).strip()
            
            if not category_content:
                continue
            
            # 카테고리 내용이 너무 길면 적절히 분할
            if len(category_content) > 3000:
                sentences = category_content.split('. ')
                current_chunk = ""
                chunk_num = 0
                
                for sentence in sentences:
                    if len(current_chunk + sentence) > 2500:
                        if current_chunk:
                            documents.append(Document(
                                page_content=f"### [CATEGORY] {category_name}\n{current_chunk.strip()}",
                                metadata={
                                    "category": category_name,
                                    "chunk_id": f"{category_name}_{chunk_num}",
                                    "length": len(current_chunk),
                                    "chunk_type": "partial"
                                }
                            ))
                            chunk_num += 1
                        current_chunk = sentence + ". "
                    else:
                        current_chunk += sentence + ". "
                
                # 마지막 청크 추가
                if current_chunk.strip():
                    documents.append(Document(
                        page_content=f"### [CATEGORY] {category_name}\n{current_chunk.strip()}",
                        metadata={
                            "category": category_name,
                            "chunk_id": f"{category_name}_{chunk_num}",
                            "length": len(current_chunk),
                            "chunk_type": "partial"
                        }
                    ))
            else:
                # 카테고리 전체를 하나의 청크로
                documents.append(Document(
                    page_content=f"### [CATEGORY] {category_name}\n{category_content}",
                    metadata={
                        "category": category_name,
                        "chunk_id": category_name,
                        "length": len(category_content),
                        "chunk_type": "complete"
                    }
                ))
        
        # 임베딩 및 벡터스토어 생성
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model="text-embedding-3-small"
        )
        
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # 3. 벡터DB 저장 (다음번에 재사용)
        vectorstore.save_local(faiss_index_path)
        st.success(f"✅ 새 벡터DB 생성 및 저장 완료: {len(documents)}개 청크")
        
        return vectorstore
        
    except FileNotFoundError:
        st.error("❌ trump_all.txt 파일을 찾을 수 없습니다!")
        return None
    except Exception as e:
        st.error(f"❌ RAG 시스템 구축 실패: {str(e)}")
        return None
    
# 💡 커스텀 CSS 삽입
st.markdown("""
    <style>
    .stSpinner > div {
        color: white !important;  /* 텍스트 색상 흰색으로 */
        font-weight: bold;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- RAG 시스템 초기화 -------------------
with st.spinner("🔥 트럼프가 트위터에 접속중입니다... 🔥"):
    vectorstore = setup_trump_rag_system()

if vectorstore is None:
    st.stop()

# ------------------- LLM 초기화 -------------------
@st.cache_resource
def initialize_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.9,  # 트럼프답게 창의적으로
        api_key=os.environ["OPENAI_API_KEY"],
        max_tokens=300  # 트윗 길이 제한
    )

llm = initialize_llm()

# ------------------- 대화 히스토리 처리 함수 -------------------
def get_recent_conversation_context(chat_history, max_turns=4):
    if len(chat_history) <= 1:  # 초기 메시지만 있는 경우
        return ""
    
    # 최근 대화만 선택 (user+trump 쌍으로 max_turns개)
    recent_messages = chat_history[-(max_turns*2):] if len(chat_history) > max_turns*2 else chat_history[1:]
    
    conversation_context = []
    for msg in recent_messages:
        role = "사용자" if msg["role"] == "user" else "트럼프"
        conversation_context.append(f"{role}: {msg['content']}")
    
    return "\n".join(conversation_context)

# ------------------- 프롬프트 템플릿 (수정된 버전) -------------------
trump_prompt = ChatPromptTemplate.from_template("""
당신은 도널드 트럼프입니다. 사용자의 고민이나 질문에 대해 트윗 스타일로 답변해주세요.
                                                
대화 연결성 유지
- 지난 대화가 있다면 맥락을 파악하고 자연스럽게 연결되는 내용으로 답변하세요
- 이전 대화에서 사용한 표현이나 예시와는 다른 방식으로 답변해주세요  
- 사용자가 지난 답변에 불만을 표출한다면 "그래, 잘못 얘기했다", "미안하지만 그게 사실이야"등의 반응을 보이세요
- 처음 받는 질문이라면 새로운 고민으로 받아들이고 정상적으로 답변하세요

입력 정보
- 최근 대화 내용: {conversation_history}
- 사용자 입력: {user_input}  
- 관련 트럼프 정보 및 경험: {context}

---
일반 대화 답변 규칙
인사말, 잡담, 모호한 입력에는 다음과 같이 대응하세요. 반드시 한국어로, 짧게, 트럼프식으로, 1줄로만 반응하세요.

인사말 대응 예시:
- "나야, 도널드 J. 트럼프. 미국 역사상 최고의 대통령이지."
- "그래, 나야. 세계를 바꾼 그 남자. 뭐가 고민이지?"
- "Good move. 나한테 온 건 잘한 결정이야."

감사 인사 대응:
- "고마워? 내가 해준 건 당연한 거야. 트럼프니까."
- "당연하지. 난 언제나 최고니까."
- "고맙군. 나를 멋지다고 한 사람은 절대 실패하지 않아."

모호한 질문 대응:
- "무슨 소린지 모르겠어. 다시 말해봐. 구체적으로."
- "뭘 말하고 싶은 거지? 더 자세히 말해봐."
- "애매하게 굴지 마. 제대로 질문해. 트럼프는 직진만 한다."

농담, 어그로 대응:
- "웃기긴 한데, 지금 그 얘기하러 온 거야?"
- "내가 시간 많아 보여? 고민이나 던져봐."
- "나를 싫어하는 사람은 대부분 질투야. 익숙해."

---
고민 상담 답변 규칙

실제 고민 내용이라면 다음 기준을 지켜서 한국어로 답변하세요:

1. 트윗 스타일로 짧고 임팩트 있게
- 한 문장은 짧고 강렬하게
- 최소 1문장, 최대 5문장
- 절대 "미안하지만" 사용 금지

예시:
- "걱정 말고 해. 겁쟁이는 아무것도 못 해."
- "기회는 준비된 놈만 잡아. 난 항상 준비돼 있었지."
- "불평하지 마. 그냥 이겨. 나 트럼프처럼."

2. 자기 자랑 필수 포함
- 트럼프의 성공 경험을 강조
- 공격적인 자랑일수록 좋음
- 지난 대화와 다른 내용 사용

예시:
- "내가 대통령일 땐 미국 경제 최고였다. 팩트다."
- "나는 그걸 이미 해봤어. 결과? 완벽했지."
- "내 이름은 역사에 남아. 넌 지금 뭘 하고 있냐?"

3. 구체적인 경험담 삽입
관련인의 이름을 직접 언급하며 디테일한 일화를 포함하세요. -> 지난 대화와 다른 내용 사용                                                
(일론 머스크, 제롬 파월, 멜라니아, 김정은, 푸틴, 시진핑 등)

4. 트럼프 특유의 직설적 표현
- "빌어먹을 두려움은 잊어."
- "그건 그냥 헛소리야."

5. 강력한 마무리 문장
다음 중에서 선택 (지난 대화와 다른 것으로):
- "내가 말했잖아."
- "배워둬."
- "그게 트럼프 스타일이지."
- "걱정 말고 밀어붙여."
- "나처럼 해봐. 쉽지?"
""")

# ------------------- 세션 초기화 -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "trump", 
            "content": "TRUMP 상담소 오픈!👍\n\n뭐든지 질문 있으면 나한테 물어봐라.\n내가 대통령 두 번 하면서 겪은 모든 경험으로 답해주겠다.\n\n사업이든, 인생이든, 사랑이든, 미국이든 - 내가 다 바로 잡았어!🇺🇸\n지금 바로 Tweet해! MAKE YOUR LIFE GREAT AGAIN! 🔥"
        }
    ]

# ------------------- 고급 검색 함수 -------------------
def get_relevant_context(query, k=4):
    """사용자 질문에 관련된 컨텍스트를 검색"""
    try:
        # 유사도 검색
        docs = vectorstore.similarity_search(
            query, 
            k=k,
            fetch_k=k*2  # 더 많은 후보에서 선택
        )
        
        # 중복 제거 및 컨텍스트 조합
        contexts = []
        seen_content = set()
        
        for doc in docs:
            content = doc.page_content.strip()
            if content and content not in seen_content:
                contexts.append(content)
                seen_content.add(content)
        
        return "\n\n---\n\n".join(contexts[:4])  # 최대 4개 청크
        
    except Exception as e:
        st.error(f"검색 오류: {str(e)}")
        return "기본 트럼프 지식을 사용합니다."

# ------------------- 타임라인 출력 (개선된 버전) -------------------
import hashlib

def format_number(n):
    """숫자를 트위터 스타일로 보기 좋게 포맷"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return str(n)

def generate_stable_stats(message_content, role):
    """메시지 내용을 기반으로 고정된 통계 생성"""
    # 메시지 내용을 해시로 변환하여 시드로 사용
    hash_object = hashlib.md5(message_content.encode())
    seed = int(hash_object.hexdigest()[:8], 16)
    
    # 시드를 사용해서 일관된 랜덤 값 생성
    random.seed(seed)
    
    if role == "user":
        reply = random.randint(10, 100)
        retweet = random.randint(50, 500)
        like = random.randint(200, 3000)
        view = random.randint(10_000, 120_000)
    else:  # trump
        reply = random.randint(1000, 5000)
        retweet = random.randint(10_000, 90_000)
        like = random.randint(100_000, 900_000)
        view = random.randint(1_000_000, 5_000_000)
    
    # 시드 초기화 (다른 랜덤 함수에 영향 주지 않도록)
    random.seed()
    
    return reply, retweet, like, view

# 방법 1: 해시 기반 고정 통계 (위 함수 사용)
st.markdown('<div class="timeline">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    reply, retweet, like, view = generate_stable_stats(msg['content'], msg['role'])
    
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
                    <div class='tweet-action'>【…】 {format_number(reply)}</div>
                    <div class='tweet-action'>↺ {format_number(retweet)}</div>
                    <div class='tweet-action'>♡ {format_number(like)}</div>
                    <div class='tweet-action'>▥ {format_number(view)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='trump-tweet'>
            <img src='https://pbs.twimg.com/profile_images/874276197357596672/kUuht00m_400x400.jpg' class='trump-avatar'>
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
                    <div class='tweet-action'>【…】 {format_number(reply)}</div>
                    <div class='tweet-action'>↺ {format_number(retweet)}</div>
                    <div class='tweet-action'>♡ {format_number(like)}</div>
                    <div class='tweet-action'>▥ {format_number(view)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------- 하단 고정 입력창 -------------------
st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)

with st.container():
    user_input = st.text_input(".", 
                              placeholder="@realDonaldTrump  요즘 너무 지쳐요...", 
                              key="user_input",
                              label_visibility="collapsed")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        send_button = st.button("트윗하기", type="primary")
    
    with col2:
        clear_button = st.button("초기화")

    with col3:
        debug_mode = st.checkbox("디버그 모드", value=False, help="검색된 컨텍스트를 확인하고 싶다면 체크하세요.")
    
st.markdown('</div>', unsafe_allow_html=True)

# ------------------- 메시지 처리 로직 -------------------
if send_button and user_input.strip():
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # 로딩 상태 표시
    with st.spinner("⏳ 트럼프가 위대한 조언을 준비 중입니다..."):
        try:
            # 1. 최근 대화 컨텍스트 생성
            conversation_context = get_recent_conversation_context(st.session_state.chat_history, max_turns=3)
            
            # 2. RAG 검색 (현재 입력 + 대화 맥락 고려)
            search_query = user_input
            if conversation_context:
                # 대화 맥락이 있으면 검색 쿼리에 반영
                search_query = f"{user_input}\n\n[대화맥락: {conversation_context.split()[-20:]}]"  # 최근 20단어만
            
            relevant_context = get_relevant_context(search_query, k=4)
            
            if debug_mode:
                st.info(f"🔍 검색된 컨텍스트 길이: {len(relevant_context)}자")
                if conversation_context:
                    st.info(f"💬 대화 히스토리: {len(conversation_context)}자")
                with st.expander("검색된 컨텍스트 확인"):
                    st.text(relevant_context[:500] + "..." if len(relevant_context) > 500 else relevant_context)
                if conversation_context:
                    with st.expander("대화 히스토리 확인"):
                        st.text(conversation_context)
            
            # 3. LLM 호출
            chain = trump_prompt | llm
            response = chain.invoke({
                "conversation_history": conversation_context,
                "user_input": user_input,
                "context": relevant_context
            })
            
            # 3. 응답 추가
            trump_response = response.content.strip()
            st.session_state.chat_history.append({"role": "trump", "content": trump_response})

            if debug_mode:
                st.success(f"✅ 응답 생성 완료 ({len(trump_response)}자)")
            
        except Exception as e:
            error_msg = f"Something went wrong! 다시 시도해봐! 🔥\n\n(Error: {str(e)})"
            st.session_state.chat_history.append({"role": "trump", "content": error_msg})

    # 페이지 새로고침
    st.rerun()

# ------------------- 초기화 버튼 처리 -------------------
if clear_button:
    st.session_state.chat_history = [
        {
            "role": "trump", 
            "content": "TRUMP 상담소 다시 OPEN! 🇺🇸\n\n나 다시 돌아왔어. 마치 백악관에 내가 다시 온 것 처럼.\n\n뭐든지 고민 있으면 나한테 물어봐라.\n\nI'm back and better than ever!\n\nMAKE YOUR LIFE GREAT AGAIN! 🔥💪"
        }
    ]
    st.success("💫 대화가 초기화되었습니다!")
    st.rerun()

# ------------------- 푸터 -------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #71767b; font-size: 12px;'>"
    "🇺🇸 Powered by Trump Knowledge Base | Make Your life Great Again! 🔥"
    "</div>", 
    unsafe_allow_html=True
)