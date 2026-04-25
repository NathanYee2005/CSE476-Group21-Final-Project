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


def self_consistency(prompt, n=5):
    answers = []
    for _ in range(n):
        result = call_model_chat_completions(prompt, temperature=0.7)
        text = (result.get("text") or "").strip()
        answers.append(text)

    return max(set(answers), key=answers.count)


def react(prompt, tools, max_steps=5):
    system = "Reply with one line: 'Action: tool[input]' or 'Final Answer: ...'"
    history = prompt

    for _ in range(max_steps):
        result = call_model_chat_completions(history, system=system)
        text = (result.get("text") or "").strip()
        history = history + "\n" + text

        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()

        match = re.search(r"Action:\s*(\w+)\[(.*?)\]", text)
        if match:
            name = match.group(1)
            arg = match.group(2)
            obs = tools[name](arg)
            history = history + "\nObservation: " + str(obs)

    return history


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
        ans = agent(q["input"])
        answers.append({"output": ans})

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)
    print(f"wrote {len(answers)} answers to {output_path}!")


def decomposition(prompt, max_parts=4):
    split = call_model_chat_completions(
        f"Break into {max_parts} numbered subproblems:\n{prompt}",
        system="Output only a numbered list of subproblems."
    )
    parts = re.findall(r"^\s*\d+[\.\)]\s*(.+)", (split.get("text") or ""), re.MULTILINE)
    if not parts:
        return call_model_chat_completions(prompt).get("text", "").strip()

    context = ""
    for part in parts:
        res = call_model_chat_completions(f"Context:\n{context}\nSubproblem: {part}")
        context += f"- {part}: {(res.get('text') or '').strip()}\n"

    final = call_model_chat_completions(
        f"Problem: {prompt}\n\nSubproblem answers:\n{context}\nFinal answer:",
        system="Reply with only the final answer—no explanation."
    )
    return (final.get("text") or "").strip()


def tool_augmented(prompt, tools=None, max_steps=4):
    system = "Reason step by step. To compute something write 'TOOL: python[code]'. When done write 'FINAL: answer'"
    history = prompt

    for _ in range(max_steps):
        result = call_model_chat_completions(history, system=system)
        text = (result.get("text") or "").strip()
        history = history + "\n" + text

        if "FINAL:" in text:
            return text.split("FINAL:")[-1].strip()

        match = re.search(r"TOOL:\s*python\[(.*?)\]", text, re.DOTALL)
        if match:
            try:
                buf = {}
                exec(match.group(1), {"math": __import__("math")}, buf)  # noqa: S102
                obs = buf.get("result", "OK")
            except Exception as e:
                obs = f"Error: {e}"
            history = history + f"\nObservation: {obs}"

    return call_model_chat_completions(
        history + "\nGive only the final answer.",
        system="Reply with only the final answer—no explanation."
    ).get("text", "").strip()


def self_refine(question: str, num_refine: int = 1):
    r1 = call_model_chat_completions(
        system="You are a helpful assistant.",
        prompt=f"Answer the question: {question}"
    )
    if not r1["ok"]:
        raise RuntimeError(f"API error: {r1['error']}")
    
    answer = r1["text"].strip()

    for i in range(num_refine):
        r2 = call_model_chat_completions(
            system="You are a critical reviewer",
            prompt=f"""
                Question:
                {question}

                Answer:
                {answer}

                Critique the answer. Point out anything that is wrong or unclear.
                """
            )
        if not r2["ok"]:
            raise RuntimeError(f"API error: {r2['error']}")
        
        critique = r2["text"].strip()

        r3 = call_model_chat_completions(
            system="You are a helpful assistant. You make a new final answer considering the previous answer and critique.",
            prompt=f"""
            Question:
            {question}

            Answer:
            {answer}

            Critique:
            {critique}

            Create a new final answer. Reply with only a single word or number.
            """
        )
        if not r3["ok"]:
            raise RuntimeError(f"API error: {r3['error']}")
        answer = r3["text"].strip()
        
    return answer

def least_to_most(question: str, max_steps: int = 5):
    r1 = call_model_chat_completions(
        system="Output only a numbered list of subproblems.",
        prompt=f"Break this into at most {max_steps} ordered subproblems: {question}"
    )
    if not r1["ok"]:
        raise RuntimeError(f"API error: {r1['error']}")

    parts = re.findall(r"^\s*\d+[\.\)]\s*(.+)", r1["text"] or "", re.MULTILINE)
    if not parts:
        parts = [(r1["text"] or "").strip()]

    context = ""

    for part in parts[:max_steps]:
        r2 = call_model_chat_completions(
            system="You are a helpful assistant.",
            prompt=f"""
                    {question}
                    Solved so far:
                    {context or "None"}
                    {part}
                    Solve only this step.
                    """
        )
        if not r2["ok"]:
            raise RuntimeError(f"API error: {r2['error']}")

        context += f"- {part}: {r2['text'].strip()}\n"

    r3 = call_model_chat_completions(
        system="You are a helpful assistant. Reply with only the final answer.",
        prompt=f"""
            {question}
            {context}
            Give the final answer.
            """
        )
    if not r3["ok"]:
        raise RuntimeError(f"API error: {r3['error']}")

    return r3["text"].strip()

def classify(question):
    system = (
        "Classify into one label: "
        "math (arithmetic with single answer), "
        "tool (needs computation or code), "
        "react (needs external lookup), "
        "decompose (independent subproblems), "
        "multistep (ordered sequential steps), "
        "refine (open-ended quality matters), "
        "reflect (tricky, may need verification), "
        "simple (everything else). "
        "Reply with only the label."
    )
    result = call_model_chat_completions(question, system=system)
    label = (result.get("text") or "").strip().lower()
    valid = {"math", "tool", "react", "decompose", "multistep", "refine", "reflect", "simple"}
    if label not in valid:
        label = "simple"
    return label


def agent(question):
    label = classify(question)
    if label == "math":
        return self_consistency(question)
    if label == "tool":
        return tool_augmented(question)
    if label == "react":
        tools = {"calc": lambda x: str(eval(x))}
        return react(question, tools)
    if label == "decompose":
        return decomposition(question)
    if label == "multistep":
        return least_to_most(question)
    if label == "refine":
        return self_refine(question)
    if label == "reflect":
        return reflection(question)["answer"]
    return cot(question)["answer"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--out", default="answers.json")
    args = parser.parse_args()

    if args.text is not None:
        print(agent(args.text))
    else:
        process_json(args.json_path, args.out)


if __name__ == "__main__":
    main()
