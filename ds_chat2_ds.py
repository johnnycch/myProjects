# input to ask question with search & recrds (no attachment) - deepssek

import os
import json
from pathlib import Path
import requests
from openai import OpenAI
from dotenv import load_dotenv

# ========== 基本設定 ==========
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("找不到 DEEPSEEK_API_KEY，請確認 .env 檔案已正確設定。")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

MODEL_NAME = "deepseek-flash"

MEMORY_FILE = Path("preferences.json")
HISTORY_FILE = Path("chat_history.json")
MAX_HISTORY_TURNS = 10

# ========== 偏好記憶 ==========
def load_preferences() -> list[str]:
    if not MEMORY_FILE.exists():
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("preferences", [])
    except (json.JSONDecodeError, OSError):
        return []

def save_preferences(prefs: list[str]) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"preferences": prefs}, f, ensure_ascii=False, indent=2)

def build_system_prompt(prefs: list[str]) -> str:
    if not prefs:
        return "你是一個樂於助人的助理。"
    pref_text = "\n".join(f"- {p}" for p in prefs)
    return (
        "你是一個樂於助人的助理。以下是我穩定不變的偏好，請在每次回答時自動遵守：\n"
        f"{pref_text}"
    )

# ========== 對話歷史 ==========
def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f).get("history", [])
        return [m for m in raw if m.get("role") in ("user", "assistant") and m.get("content")]
    except (json.JSONDecodeError, OSError):
        return []

def save_history(history: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({"history": history}, f, ensure_ascii=False, indent=2)

def trim_history(history: list[dict]) -> list[dict]:
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(history) > max_msgs:
        return history[-max_msgs:]
    return history

def history_turn_count(history: list[dict]) -> int:
    return len(history) // 2

# ========== 搜尋工具 (Function Calling) ==========
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "當需要查詢即時資訊(如天氣、新聞、股價、最新事件等)時，使用此工具進行網絡搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索的關鍵字或完整問題，例如：'香港今日天氣' 或 '恒生指數最新走勢'。",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

def execute_search_tool(query: str) -> str:
    """實際執行搜索。這裡使用 DeepSeek 官方 /v1/tools/search 端點。"""
    print(f"🔍 偵測到需要即時資訊，正在為你搜索：{query} ...")
    try:
        # 注意：DeepSeek 的官方 Search API 可能需要特定的權限或計費。
        # 如果這裡報錯，可以替換為其他搜索 API (如 DuckDuckGo Search)
        resp = requests.post(
            "https://api.deepseek.com/v1/tools/search",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"query": query, "count": 3},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return "找不到相關的即時資訊。"
        formatted = "\n\n".join(
            [f"標題：{r.get('title', '')}\n摘要：{r.get('summary', '')}" for r in results]
        )
        return formatted
    except Exception as e:
        return f"搜索執行失敗，請根據你現有的知識回答，並告知用戶搜索遇到問題。錯誤：{e}"

# ========== 對話核心 ==========
def ask_deepseek(question: str, prefs: list[str], history: list[dict]):
    """送出問題，處理工具調用，回傳 (答案, 更新後的歷史)。"""
    messages = [{"role": "system", "content": build_system_prompt(prefs)}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    # 第一次呼叫：模型決定是否要使用工具
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
    )
    response_message = response.choices[0].message

    # 檢查模型是否要求調用工具
    if response_message.tool_calls:
        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "search_web":
                tool_result = execute_search_tool(function_args["query"])
            else:
                tool_result = f"未知的工具：{function_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        # 第二次呼叫：模型根據工具結果生成最終答案
        second_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        answer = second_response.choices[0].message.content
    else:
        # 模型不需要工具，直接回答
        answer = response_message.content

    # 成功後才寫入歷史(只記錄最終問答，不記錄中間的 tool calls，避免歷史膨脹)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    history = trim_history(history)
    save_history(history)

    return answer, history

# ========== 管理介面 ==========
def show_preferences(prefs: list[str]) -> None:
    print("\n=== 目前的偏好記憶 ===")
    if not prefs:
        print("(尚無任何偏好記憶)")
    else:
        for i, p in enumerate(prefs, 1):
            print(f"{i}. {p}")
    print("======================\n")

def manage_preferences() -> list[str]:
    prefs = load_preferences()
    while True:
        print("\n【偏好記憶管理】")
        print("1. 檢視所有偏好")
        print("2. 新增偏好")
        print("3. 刪除偏好(依編號)")
        print("4. 清空所有偏好")
        print("5. 返回主選單")
        choice = input("請選擇 (1-5): ").strip()

        if choice == "1":
            show_preferences(prefs)
        elif choice == "2":
            new_pref = input("請輸入新的偏好(例如：請用繁體中文回答)：").strip()
            if new_pref:
                prefs.append(new_pref)
                save_preferences(prefs)
                print("✅ 已新增偏好。")
        elif choice == "3":
            if not prefs:
                print("目前沒有偏好可刪除。")
                continue
            show_preferences(prefs)
            try:
                idx = int(input("請輸入要刪除的編號: ")) - 1
                if 0 <= idx < len(prefs):
                    removed = prefs.pop(idx)
                    save_preferences(prefs)
                    print(f"✅ 已刪除：{removed}")
            except ValueError:
                print("❌ 請輸入數字。")
        elif choice == "4":
            if input("確定要清空所有偏好嗎？(y/N): ").strip().lower() == "y":
                prefs = []
                save_preferences(prefs)
                print("✅ 已清空所有偏好。")
        elif choice == "5":
            return prefs

def show_history(history: list[dict]) -> None:
    print("\n=== 最近對話歷史 ===")
    if not history:
        print("(尚無任何對話歷史)")
    else:
        turn = 1
        for i in range(0, len(history) - 1, 2):
            q = history[i]["content"]
            a = history[i + 1]["content"]
            print(f"\n── 第 {turn} 輪 ──")
            print(f"🙋 你：{q}")
            print(f"🤖 DeepSeek：{a}")
            turn += 1
    print(f"\n(共 {history_turn_count(history)} 輪，上限 {MAX_HISTORY_TURNS} 輪)\n")

def manage_history() -> list[dict]:
    history = load_history()
    while True:
        print("\n【對話歷史管理】")
        print("1. 檢視最近對話")
        print("2. 清除所有對話歷史")
        print("3. 返回主選單")
        choice = input("請選擇 (1-3): ").strip()

        if choice == "1":
            show_history(history)
        elif choice == "2":
            if input("確定要清除所有對話歷史嗎？(y/N): ").strip().lower() == "y":
                history = []
                save_history(history)
                print("✅ 已清除對話歷史。")
        elif choice == "3":
            return history

# ========== 主程式 ==========
def main():
    print("🤖 DeepSeek 文字題問答助理(偏好記憶＋對話歷史＋動態聯網)")
    prefs = load_preferences()
    history = load_history()

    if prefs:
        print("\n已載入偏好記憶：")
        for p in prefs:
            print(f"  • {p}")
    if history:
        print(f"已載入 {history_turn_count(history)} 輪最近對話。")

    while True:
        print("\n【主選單】")
        print("1. 輸入文字題提問")
        print("2. 管理偏好記憶")
        print("3. 管理對話歷史")
        print("4. 離開")

        choice = input("請選擇 (1-4): ").strip()

        if choice == "1":
            question = input("\n請輸入你的問題：").strip()
            if not question:
                continue
            print("\n⏳ 正在思考與查詢中...\n")
            try:
                answer, history = ask_deepseek(question, prefs, history)
                print("=== DeepSeek 回答 ===")
                print(answer)
                print("======================")
            except Exception as e:
                print(f"❌ 發生錯誤：{e}")
        elif choice == "2":
            prefs = manage_preferences()
        elif choice == "3":
            history = manage_history()
        elif choice == "4":
            print("再見！")
            break

if __name__ == "__main__":
    main()