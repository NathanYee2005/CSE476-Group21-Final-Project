import os, json, textwrap, re, time
import requests

API_KEY  = os.getenv("OPENAI_API_KEY", "CREATE FROM Voyager Portal")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")  
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")              

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}


def cot(question, temperature=0.0):
    system = (
        "Think step by step. Show your reasoning, then on the final line write:\n"
        "Final answer: <your answer>"
    )
    result = call_model_chat_completions(question, system=system, temperature=temperature)
    text = (result.get("text") or "").strip()
    if "Final answer:" in text:
        answer = text.rsplit("Final answer:", 1)[-1].strip()
    else:
        answer = text
    return {"reasoning": text, "answer": answer}


def reflection(question, max_retries=2):
    attempt = cot(question)
    history = []

    for _ in range(max_retries):
        critique_prompt = (
            f"Question: {question}\n\n"
            f"My reasoning: {attempt['reasoning']}\n"
            f"My answer: {attempt['answer']}\n\n"
            "If this answer is correct, reply with exactly: CORRECT.\n"
            "Otherwise, in one sentence, say what is wrong."
        )
        critique = (call_model_chat_completions(critique_prompt).get("text") or "").strip()

        if critique.upper().startswith("CORRECT"):
            break

        history.append(critique)
        retry_prompt = (
            f"{question}\n\n"
            "Previous attempts had these issues:\n- " + "\n- ".join(history) +
            "\n\nTry again, avoiding those issues."
        )
        attempt = cot(retry_prompt)

    return {"answer": attempt["answer"], "reasoning": attempt["reasoning"], "critiques": history}


def process_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fp:
        questions = json.load(fp)

    answers = []
    for q in questions:
        result = call_model_chat_completions(q["input"])
        ans = (result.get("text") or "").strip()
        answers.append({"output": ans})

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)
    print(f"wrote {len(answers)} answers to {output_path}!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--out", default="answers.json")
    args = parser.parse_args()

    if args.text is not None:
        result = call_model_chat_completions(args.text)
        print((result.get("text") or "").strip())
    else:
        process_json(args.json_path, args.out)


if __name__ == "__main__":
    main()
