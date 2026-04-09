import json
from collections import Counter
from config import DATASET_PATH

def analyze_dataset(file_path):
    """Analyze the emotional dataset for variety and distribution"""
    print(f"Analyzing dataset: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total = len(data)
    print(f"Total conversations: {total}")
    
    # 1. Conversation length analysis
    lengths = []
    roles = Counter()
    
    for convo in data:
        messages = convo.get("conversations", [])
        lengths.append(len(messages))
        for msg in messages:
            roles[msg["role"]] += 1
            
    avg_len = sum(lengths) / total
    print(f"Average rounds per conversation: {avg_len / 2:.2f}")
    print(f"Role distribution: {dict(roles)}")
    
    # 2. Key phrase spotting (optional, can help detect repetition)
    phrases = [
        "其实挺难过的", "对不起", "撒个娇", "AI小助手", "程序"
    ]
    
    print("\nPhrase Check (Frequency):")
    for phrase in phrases:
        count = sum(1 for convo in data if any(phrase in msg["content"] for msg in convo["conversations"]))
        print(f"- '{phrase}': {count} conversations ({(count/total)*100:.1f}%)")

    # 3. Quick check for prohibited behavior
    ai_mentions = sum(1 for convo in data if any("人工智能" in msg["content"].lower() or "大模型" in msg["content"].lower() for msg in convo["conversations"]))
    if ai_mentions > 0:
        print(f"\n[WARNING] {ai_mentions} conversations mention 'AI/LLM' - this should be minimized to maintain character personality.")

if __name__ == "__main__":
    analyze_dataset(DATASET_PATH)
