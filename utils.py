import re

def extract_emotions(text):
    emotions = {
        "기쁨": ["기뻐", "즐거", "좋아", "행복"],
        "슬픔": ["슬퍼", "우울", "눈물", "상실"],
        "분노": ["화나", "짜증", "열받", "분노"],
        "불안": ["불안", "걱정", "초조", "긴장"],
        "혼란": ["헷갈", "혼란", "정신없"]
    }
    found = set()
    for label, keywords in emotions.items():
        for kw in keywords:
            if re.search(kw, text):
                found.add(label)
    return list(found) if found else ["알 수 없음"]
