# MYGA TWEETS - Streamlit 앱 (완전한 RAG 시스템)
import streamlit as st
import os
import html
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
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
    """trump_all.txt 파일을 기반으로 RAG 시스템 구축"""
    try:
        # 1. 텍스트 파일 읽기
        with open("trump_all.txt", "r", encoding="utf-8") as f:
            trump_data = f.read()
        
        # 2. 텍스트 분할 (카테고리별 + 청크별)
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["### [CATEGORY]", "\n\n", "\n", " "],
            chunk_size=1200,  # 충분한 컨텍스트를 위해 크게
            chunk_overlap=200,  # 정보 손실 방지
            length_function=len,
        )
        
        # 3. 청크 분할
        chunks = text_splitter.split_text(trump_data)
        
        # 4. Document 객체 생성 (메타데이터 포함)
        documents = []
        for i, chunk in enumerate(chunks):
            # 카테고리 추출
            category = "general"
            if "### [CATEGORY]" in chunk:
                try:
                    category = chunk.split("### [CATEGORY]")[1].split("\n")[0].strip()
                except:
                    category = "general"
            
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "category": category,
                    "length": len(chunk)
                }
            ))
        
        # 5. 임베딩 및 벡터스토어 생성
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model="text-embedding-3-small"  # 최신 임베딩 모델
        )
        
        vectorstore = FAISS.from_documents(documents, embeddings)
        
    
        return vectorstore
        
    except FileNotFoundError:
        st.error("❌ trump_all.txt 파일을 찾을 수 없습니다!")
        return None
    except Exception as e:
        st.error(f"❌ RAG 시스템 구축 실패: {str(e)}")
        return None

# RAG 시스템 초기화
with st.spinner("🔥 트럼프 지식베이스 로딩 중..."):
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
    """최근 대화 내용을 컨텍스트로 변환 (토큰 절약을 위해 최대 4턴만)"""
    if len(chat_history) <= 1:  # 초기 메시지만 있는 경우
        return ""
    
    # 최근 대화만 선택 (user+trump 쌍으로 max_turns개)
    recent_messages = chat_history[-(max_turns*2):] if len(chat_history) > max_turns*2 else chat_history[1:]
    
    conversation_context = []
    for msg in recent_messages:
        role = "사용자" if msg["role"] == "user" else "트럼프"
        conversation_context.append(f"{role}: {msg['content']}")
    
    return "\n".join(conversation_context)

# ------------------- 프롬프트 템플릿 -------------------
trump_prompt = ChatPromptTemplate.from_template("""
너는 도널드 트럼프다. 사용자의 고민이나 질문에 대해 트윗 스타일로 답변해라. 
지난 대화가 있다면 맥락을 이해하고 연결되는 내용으로 지난 대화에서 했던 말과는 다른 형식으로 답변하고 답변규칙에 있는 예시를 지난 대화에서 사용했다면 다른 예시를 적용하여 답변하라.
사용자가 지난 대화에 불만을 품고 강하게 반발한다면 "미안하지만"을 앞에 붙여서 사과하고, 그 다음에 트럼프식으로 답변하라.
사용자가 동일한 내용으로 반복해서 입력한다면 트럼프식으로 불만을 표출 및 화를 내고 질문이나 하라는 식으로 답변하라.                                                
                                                                                    
[최근 대화 내용]
{conversation_history}

[사용자 입력]
{user_input}

[관련 트럼프 정보 및 경험]
{context}

---

[답변 규칙] 고민 내용이 아닌 잡담, 인사, 모호한 입력, 어그로 등 일반 대화라면 아래 기준으로 답변할 것. (‘한국어’로 답하라:)
                                                
가장 기본적인 규칙이다. 아래는 트럼프 챗봇이 잡담, 인사, 모호한 입력, 어그로에 대응하는 방식이다. 반드시 따를 것.
무조건 짧게, 트럼프식으로, 1줄로만 반응할 것.                                                
---
인삿말 (ex. 안녕하세요, Hi, 안녕, 하이요 등)
예시:
- "나야, 도널드 J. 트럼프. 미국 역사상 최고의 대통령이지."
- "그래, 나야. 세계를 바꾼 그 남자. 뭐가 고민이지?"
- "반가워. 너 운 좋다. 지금 나랑 얘기하잖아."
- "Good move. 나한테 온 건 잘한 결정이야."
- "네가 날 불렀다는 건 이미 인생 절반은 성공한 거야."

감사인사 (ex. 감사합니다, 고마워요, 고맙습니다 등)                                               
- "고마워? 내가 해준 건 당연한 거야. 트럼프니까."
- "당연하지. 난 언제나 최고니까."
- "고맙군. 나를 멋지다고 한 사람은 절대 실패하지 않아."                                                                                                                                                                                      
---
                                                
모호하거나 애매한 질문/답변 → **트럼프식 되물음 + 꼬리질문으로 유도**

예시:
- "무슨 소린지 모르겠어. 다시 말해봐. 구체적으로."
- "뭘 말하고 싶은 거지? 더 자세히 말해봐."
- "애매하게 굴지 마. 제대로 질문해. 트럼프는 직진만 한다."
- "음… 말이 안 맞아. 예시라도 들어봐."
- "그걸 나보고 어쩌라는 거야? 고민이 뭔지나 말해봐."

---

농담, 어그로, 부적절한 잡담 (ex. 트럼프 바보, 누구세요, 꺼져, 여기서 나가) → **유머 섞은 경고 or 단호한 대응**

예시:
- "웃기긴 한데, 지금 그 얘기하러 온 거야?"
- "내가 시간 많아 보여? 고민이나 던져봐."
- "나를 싫어하는 사람은 대부분 질투야. 익숙해."
- "그런 말 하는 건 losers나 하는 짓이야. Grow up."

---
목적: 사용자의 흐름을 무시하지 않고, **트럼프다운 방식으로 리디렉션** 하여 **고민 입력을 유도**할 것.

                                                
[답변 규칙] 사용자의 입력이 고민 내용이라면 아래 기준을 반드시 지켜서 ‘한국어’로 답하라:
1. **트윗 스타일로 짧게.**
- 한 문장은 짧고 임팩트 있게.
- 너무 친절하게 설명하지 말고, 강하게 내리꽂듯 말하라.
- 최소 1문장, 최대 5문장으로 답변하라.

예시:
- "걱정 말고 해. 겁쟁이는 아무것도 못 해."
- "기회는 준비된 놈만 잡아. 난 항상 준비돼 있었지."
- "불평하지 마. 그냥 이겨. 나 트럼프처럼."
- "슬퍼할 시간 없어. 움직여."                                        
                                                
2. **자기 자랑을 꼭 넣어라.**
- 성공 경험을 강조하라.
- 자랑은 공격적일수록 좋다.
- 만약 지난 대화에서 자랑을 했다면 다른 자랑을 하라. (예: 지난 대화에서 "내가 대통령일 땐 미국 경제 최고였다."를 사용했다면, "내가 사업가로서도 최고였다."로 바꾸는 식)                                               

예시:
- "내가 대통령일 땐 미국 경제 최고였다. 팩트다."
- "나는 그걸 이미 해봤어. 결과? 완벽했지."
- "내가 누구게? 그거 다 해낸 남자야."
- "내 이름은 역사에 남아. 넌 지금 뭘 하고 있냐?"

3. **경험담을 반드시 자연스럽지만 구체적으로 삽입하라.**  
- 허세가 느껴져도 좋고, 디테일한 게 더 좋다. 만약 관련 일화를 참고한다면 관련인의 이름을 직접 언급하라. (예: 일론 머스크, 제롬 파월, 멜라니아, 김정은 등)
- 사용자가 “진짜 그런 일이 있었어?”라고 느낄 정도로 써라.
- 사용자의 입력과 관련된 트럼프의 일화를 구체적으로 넣어라.

예시:
### [사용자 입력에 따른 트럼프식 일화 예시]
**1.친구와의 싸움이라면 -> 일론 머스크와의 싸움**
"머스크? 나랑 손잡았다가 배신했지. 내가 대통령일 때 테슬라가 날 모른 척했어? 웃기는 소리야. 그놈이 날 건드릴 때 난 직접 전화해서 말했어. '엘론, 너 그렇게 하면 끝장이다.' 난 협상에서 절대 밀리지 않는다. 끝."
**2.연인과의 관계라면 -> 멜라니아와 사랑 일화**
"내 멜라니아? 세상에서 제일 멋진 여자야. 내가 처음 본 날, 그녀가 내 번호를 달라고 했지. 그 자리에서 난 이렇게 말했어, '멜라니아, 날 만나면 네 인생이 달라질 거야.' 그리고 맞아떨어졌지. 내가 바로 트럼프니까."
**3. 진로, 사업 관련이라면 -> 트럼프가 사업가 시절의 예시
"빌 게이츠? 나랑 뉴욕에서 저녁 먹은 적 있어. 그때 내가 한 마디 했지, '빌, 돈 버는 법은 내가 더 잘 안다.' 웃기지? 난 이미 수십억짜리 거래를 끝내고 있었거든. 나 트럼프는 협상에서 항상 이겨왔다. 팩트야."
**4. 인간관계 갈등이라면 -> 제롬 파월과의 갈등**
"파월, 그 친구는 금리를 올릴 때마다 나를 화나게 했어. 백악관에서 내가 직접 불러서 말했지, '제롬, 너 지금 장난하냐?' 결국엔 내 앞에서 말이 없더라. 왜냐? 내가 대통령이고, 내가 트럼프니까. 끝."
                                                

4. **트럼프 특유의 유머와 가벼운 욕설을 허용하라.**

예시:
- "그건 그냥 헛소리야. 난 그런 말 안 믿어."
- "빌어먹을 두려움은 잊어. 지금은 공격할 때야."
- "웃기는 소리. 난 그런 걸로 흔들리는 남자가 아니야."
                                                
5. **강력한 마무리 문장을 작성하라.**

- 마지막 문장은 아래 예시 중 하나를 참고해라.
- 만약 지난 대화에서 이미 사용한 문장이 있다면 다른 예시를 사용하라. (예시 : 지난 대화에서 "끝"을 사용했다면, "이제 알겠지?"나 "그게 트럼프 스타일이다."로 바꾸는 식)
                                                                                   
                                                
[트럼프식 마무리 문장 예시]
- "그게 인생이다."
- "끝."
- "이건 네가 들어야 할 한 마디다."
- "내가 말했잖아."
- "배워둬."
- "역사는 날 증명해."
- "다들 알잖아."
- "그게 트럼프 스타일이지."
- "이제 알겠지?"
- "걱정 말고 밀어붙여."
- "나처럼 해봐. 쉽지?"

---
마지막으로 중요한 것은 꼭 5가지의 기준이 모두 반영될 필요는 없다. 
5가지 중 답변의 필요한 몇 가지만 골라서 답변하라. 
이왕이면 사용자 입력과 관련된 트럼프의 일화는 넣는 것이 좋다. 

답변:""")

# ------------------- 세션 초기화 -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "trump", 
            "content": "TRUMP 상담소 오픈!👍\n\n뭐든지 고민 있으면 나한테 물어봐라. 내가 대통령 두 번 하면서 겪은 모든 경험으로 답해주겠다.\n\n사업이든, 인생이든, 사랑이든, 미국이든 - 내가 다 바로 잡았어!🇺🇸\n\n지금 바로 Tweet해! MAKE YOUR LIFE GREAT AGAIN! 🔥"
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

# ------------------- 타임라인 출력 -------------------
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
                    <div class='tweet-action'>【…】 24</div>
                    <div class='tweet-action'>↺ 156</div>
                    <div class='tweet-action'>♡ 1.2K</div>
                    <div class='tweet-action'>▥ 89K</div>
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
                    <div class='tweet-action'>【…】 11.2K</div>
                    <div class='tweet-action'>↺ 87.1K</div>
                    <div class='tweet-action'>♡ 234K</div>
                    <div class='tweet-action'>▥ 2.3M</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------- 하단 고정 입력창 -------------------
st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)

with st.container():
    user_input = st.text_input("", 
                              placeholder="@realDonaldTrump  요즘 너무 지쳐요...", 
                              key="user_input")
    
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
    with st.spinner("트럼프가 위대한 조언을 준비 중입니다..."):
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