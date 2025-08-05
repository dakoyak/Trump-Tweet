# MYGA TWEETS - 트럼프 트위터 상담소 🇺🇸

도널드 트럼프 스타일로 답변해주는 AI 상담 웹앱입니다. RAG(Retrieval-Augmented Generation) 시스템을 활용하여 트럼프의 실제 발언과 경험을 바탕으로 현실적인 답변을 제공합니다.

## 🌟 주요 기능

### 🎯 핵심 특징
- **트위터 UI**: 실제 트위터와 동일한 디자인과 사용자 경험
- **RAG 시스템**: 트럼프의 실제 발언 데이터베이스를 기반으로 한 맞춤형 답변
- **대화 연결성**: 이전 대화 맥락을 파악하여 자연스러운 대화 진행
- **실시간 상호작용**: 즉시 응답하는 채팅 형태의 상담

### 📱 사용자 인터페이스
- 트럼프 공식 트위터 프로필 복제
- 트윗 스타일 메시지 표시
- 고정된 하단 입력창
- 좋아요, 리트윗 등 가짜 통계 표시

## 🛠️ 기술 스택

### Frontend
- **Streamlit**: 웹 애플리케이션 프레임워크
- **HTML/CSS**: 트위터 스타일 커스텀 디자인
- **JavaScript**: 동적 UI 요소

### Backend & AI
- **OpenAI GPT-4o-mini**: 대화 생성 모델
- **LangChain**: AI 워크플로우 관리
- **FAISS**: 벡터 데이터베이스 (유사도 검색)
- **OpenAI Embeddings**: 텍스트 임베딩

### 데이터 처리
- **RAG (Retrieval-Augmented Generation)**: 지식 기반 답변 생성
- **Vector Search**: 관련 컨텍스트 자동 검색
- **Text Splitting**: 효율적인 문서 분할

## 📋 설치 및 실행

### 1. 환경 설정
```bash
# 저장소 클론
git clone <repository-url>
cd myga-tweets

# 의존성 설치
pip install streamlit langchain langchain-community langchain-openai faiss-cpu python-dotenv
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 OpenAI API 키를 추가하세요:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 데이터 파일 준비
프로젝트 루트에 `trump_all.txt` 파일이 필요합니다. 이 파일은 다음 형식으로 구성되어야 합니다:
```
### [CATEGORY] 정치
트럼프의 정치 관련 발언들...

### [CATEGORY] 사업
트럼프의 사업 관련 발언들...

### [CATEGORY] 개인생활
트럼프의 개인적 경험들...
```

### 4. 애플리케이션 실행
```bash
streamlit run app.py
```

## 🔧 주요 구성 요소

### RAG 시스템
- **데이터 로딩**: `trump_all.txt`에서 카테고리별 데이터 로드
- **문서 분할**: 카테고리별로 적절한 크기로 청크 분할
- **벡터화**: OpenAI 임베딩을 사용한 텍스트 벡터화
- **저장/로드**: FAISS 인덱스 캐싱으로 빠른 재실행

### 대화 시스템
- **컨텍스트 관리**: 최근 4턴의 대화 히스토리 유지
- **프롬프트 엔지니어링**: 트럼프 스타일 답변을 위한 상세한 프롬프트
- **응답 생성**: GPT-4o-mini를 활용한 개성 있는 답변

### UI/UX
- **반응형 디자인**: 모바일과 데스크톱 모두 지원
- **실시간 업데이트**: 메시지 전송 후 즉시 화면 갱신
- **통계 생성**: 메시지 내용 기반 일관된 가짜 통계

## 🎮 사용법

1. **웹페이지 접속**: 애플리케이션 실행 후 제공되는 URL 접속
2. **질문 입력**: 하단 입력창에 고민이나 질문 입력
3. **트윗 전송**: "트윗하기" 버튼 클릭
4. **답변 확인**: 트럼프 스타일의 조언 확인
5. **대화 계속**: 연속적인 대화로 더 깊은 상담 가능

### 예시 대화
```
사용자: "요즘 취업이 안 돼서 너무 스트레스예요..."
트럼프: "취업? 내가 대통령일 때 실업률 최저였어. 포기하지 마. 
       기회는 준비된 놈만 잡아. 나처럼 해봐. 쉽지?"
```

## ⚙️ 설정 옵션

### 디버그 모드
- 검색된 컨텍스트 확인 가능
- 대화 히스토리 추적
- 응답 생성 과정 모니터링

### 대화 관리
- **초기화**: 대화 히스토리 완전 삭제
- **컨텍스트 제한**: 최대 4턴까지 대화 맥락 유지
- **메시지 길이**: 트윗 스타일로 300토큰 제한

## 🔒 보안 및 주의사항

- OpenAI API 키는 반드시 환경 변수로 관리
- FAISS 인덱스 파일의 `allow_dangerous_deserialization=True` 설정은 로컬 환경에서만 사용
- 상용 환경에서는 추가적인 보안 검토 필요

## 🐛 문제 해결

### 일반적인 오류
1. **API 키 오류**: `.env` 파일의 OpenAI API 키 확인
2. **파일 없음 오류**: `trump_all.txt` 파일 존재 여부 확인
3. **메모리 부족**: 벡터 데이터베이스 크기 조정 필요

### 성능 최적화
- 벡터 인덱스 캐싱으로 재실행 시간 단축
- 문서 청크 크기 조정으로 검색 품질 향상
- GPT 모델 온도 조정으로 답변 스타일 변경 가능

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다. 상업적 사용 시 관련 라이선스를 확인하시기 바랍니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**MAKE YOUR LIFE GREAT AGAIN! 🔥🇺🇸**
