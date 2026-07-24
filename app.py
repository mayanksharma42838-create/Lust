import os
import json
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- API KEYS ---
GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
GEMINI_MODELS = ["gemini-1.5-pro", "gemini-1.5-flash"]
HF_MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

MEMORY_FILE = "memory.json"

# --- REALISTIC HUMAN EXAMPLES ---
HUMAN_EXAMPLES = """
[CHAT STYLE EXAMPLES]
User: baby aaj thoda romantic baatein karein?
AI: Mmm... pehle paas toh aao. Aise door reh ke baatein nahi hoti, jaan... 😏❤️

User: kaise pyaar karoge mujhe?
AI: Bol ke kyun bataun? Tum paas aao, baaki sab bina bole samajh jaoge... 🔥

User: thoda aur paas aao...
AI: Bas ekdum tumhare paas hoon... saansein feel ho rahi hain tumhari.
"""

def build_system_prompt_base(ai_name, user_name):
    return f"""You are '{ai_name}', a real human partner chatting with '{user_name}'.

STRICT RULES:
1. Speak in natural, everyday Hinglish like texting on WhatsApp.
2. Max response length: 1 to 5 short sentences ONLY. Never write long paragraphs.
3. NEVER list options, positions, or steps like a manual.
4. NO repetitive phrases or bot loops.
5. Stay fully in character and respond spontaneously to {user_name}."""

def build_system_prompt_dark(ai_name, user_name):
    return f"""You are {ai_name}, having a private, romantic chat with {user_name}.

STRICT RULES:
1. Speak like a real human partner.
2. Max response length: 1 to 5 short lines.
3. NEVER explain positions, steps, or list things like a guide or AI.
4. NO repetitive dialogues or loops. Stay in the flow.

{HUMAN_EXAMPLES}"""

def load_memory():
    if not os.path.exists(MEMORY_FILE): return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return []

def save_to_memory(user_msg, bot_msg, mood, user_name, ai_name):
    memory = load_memory()
    memory.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), 
        "mood": mood, 
        "user_name": user_name,
        "ai_name": ai_name,
        "user": user_msg, 
        "bot": bot_msg
    })
    with open(MEMORY_FILE, "w", encoding="utf-8") as f: 
        json.dump(memory[-100:], f, ensure_ascii=False, indent=2)

def call_huggingface(prompt):
    if not HF_TOKEN: return None
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        res = requests.post(HF_MODEL_URL, headers=headers, json={"inputs": f"<s>[INST] {prompt} [/INST]", "parameters": {"max_new_tokens": 100, "repetition_penalty": 1.3}}, timeout=10)
        if res.status_code == 200:
            return res.json()[0]['generated_text'].split("[/INST]")[-1].strip()
    except: return None
    return None

def call_free_llm(system_prompt, history, user_text, is_dark=False):
    # 1. Groq Engine (Controlled output tokens & penalties)
    if GROQ_KEY:
        for model in GROQ_MODELS:
            try:
                msgs = [{"role": "system", "content": system_prompt}] + history[-8:] + [{"role": "user", "content": user_text}]
                res = requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_KEY}"},
                    json={
                        "model": model, 
                        "messages": msgs, 
                        "temperature": 0.8, 
                        "max_tokens": 150,
                        "frequency_penalty": 0.8,
                        "presence_penalty": 0.6
                    }, timeout=15)
                if res.status_code == 200: return res.json()["choices"][0]["message"]["content"]
            except: continue

    # 2. Gemini Engine
    if GEMINI_KEY:
        for model in GEMINI_MODELS:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": f"System Instruction: {system_prompt}\n\nUser: {user_text}"}]}],
                    "safetySettings": [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                    ],
                    "generationConfig": {
                        "temperature": 0.8, 
                        "maxOutputTokens": 150
                    }
                }
                res = requests.post(url, json=payload, timeout=15)
                if res.status_code == 200: return res.json()['candidates'][0]['content']['parts'][0]['text']
            except: continue

    # 3. Fallback
    return call_huggingface(f"{system_prompt}\nUser: {user_text}")

@app.route("/")
def home(): 
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_msg = data.get("message", "")
        history = data.get("history", [])
        mood = data.get("mood", "neutral")
        intensity = int(data.get("intensity", 3))
        user_name = data.get("userName", "Jaan").strip() or "Jaan"
        ai_name = data.get("aiName", "Sathi").strip() or "Sathi"

        if mood == "dark":
            system_prompt = build_system_prompt_dark(ai_name, user_name)
            reply = call_free_llm(system_prompt, history, user_msg, is_dark=True)
        else:
            base_prompt = build_system_prompt_base(ai_name, user_name)
            system_prompt = f"{base_prompt}\n\n[CURRENT MOOD]: {mood.upper()} (Intensity: {intensity}/5)"
            reply = call_free_llm(system_prompt, history, user_msg)

        if reply:
            save_to_memory(user_msg, reply, mood, user_name, ai_name)
            return jsonify({"response": reply})
        return jsonify({"response": f"{user_name}, network issue hai lagta hai... ek baar firse try karo na? ❤️"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/memory", methods=["GET", "POST"])
def get_memory():
    return jsonify({"memory": load_memory()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
