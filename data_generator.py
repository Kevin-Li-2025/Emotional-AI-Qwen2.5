import json
import time
import requests
from config import *

def generate_conversation(scenario_name):
    """Generate a single emotional conversation scenario using DeepSeek API"""
    
    # Prompt construction based on scenario and character description
    system_prompt = f"{CHARACTER_DESCRIPTION}\n\n当前场景：{scenario_name}\n请生成一段自然的、多轮（3-5轮）的情感对话。以JSON格式输出，包含'conversations'列表，每项有'role' (user/assistant) 和 'content'。"
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "开始一段对话"}
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=data)
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        return json.loads(content)
    except Exception as e:
        print(f"Error generating for {scenario_name}: {e}")
        return None

def main():
    """Main loop to generate the full dataset based on SCENARIOS in config.py"""
    all_data = []
    print(f"Starting data generation (Target: {sum(SCENARIOS.values())} conversations)...")
    
    for scenario, count in SCENARIOS.items():
        print(f"Generating {count} conversations for scenario: {scenario}")
        for i in range(count):
            convo = generate_conversation(scenario)
            if convo:
                all_data.append(convo)
                if len(all_data) % 10 == 0:
                    print(f"Progress: {len(all_data)} conversations generated...")
                    # Immediate save to prevent data loss
                    with open(DATASET_PATH, "w", encoding="utf-8") as f:
                        json.dump(all_data, f, ensure_ascii=False, indent=2)
            time.sleep(0.5) # Rate limiting

    print(f"Success! Saved {len(all_data)} conversations to {DATASET_PATH}")

if __name__ == "__main__":
    main()
