import os, json, textwrap, re, time, math
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

API_KEY  = os.getenv("OPENAI_API_KEY", "CREATE FROM Voyager Portal")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL    = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

LLM_CALLS = 0


def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    global LLM_CALLS
    LLM_CALLS += 1
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
        "max_tokens": 2048,
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


# %% Simple normalization and evaluation helpers
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    # Remove surrounding punctuation and extra whitespace
    s = re.sub(r"[^\w\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Map common synonyms used in these tests
    synonyms = {
        "unchanged": "stay the same",
        "no change": "stay the same",
        "same": "stay the same",
        "second place": "second",
        "2nd": "second",
        "first place": "first",
        "third place": "third",
    }
    return synonyms.get(s, s)

def extract_number(s: str):
    # Returns first number occurrence as string if found, else None
    if not s:
        return None
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return m.group(0) if m else None


def _extract_boxed(text: str) -> str | None:
    #fix extraction to find total {} instead of stopping on first one and returning nothing
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    inner_start = start + len("\\boxed{")
    open_braces = 1
    for i in range(inner_start, len(text)):
        if text[i] == "{":
            open_braces += 1
        elif text[i] == "}":
            open_braces -= 1
            if open_braces == 0:
                return text[inner_start:i]
    return None


def _strip_answer_markers(text: str) -> str:
    boxed = _extract_boxed(text)
    if boxed is not None:
        text = boxed
    text = re.sub(r"^\**\s*(final answer|answer)\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip("*`\"' .,")

#runs python code
def _tool_python(code: str) -> str:
    scope = {"math": math}
    exec(code, scope)
    return str(scope.get("result", "OK"))


#evaluates simple math expressions
def _tool_calc(expr: str) -> str:
    return str(eval(expr, {"math": math}))

TOOLS = {"python": _tool_python, "calc": _tool_calc}


def self_consistency(prompt, n=5, _escalate=True):
    raw_answers = []
    for _ in range(n):
        result = call_model_chat_completions(prompt, temperature=0.7)
        raw_answers.append((result.get("text") or "").strip())

    normalized = []
    for answer in raw_answers:
        stripped = _strip_answer_markers(answer)
        number = extract_number(stripped)
        normalized.append(number if number is not None else normalize_text(stripped))

    valid = [v for v in normalized if v]
    if not valid:
        return raw_answers[0] if raw_answers else ""

    winner = max(set(valid), key=valid.count)
    winner_count = valid.count(winner)

    #escalate to reflection if the winner did not get over 50% votes
    if _escalate and winner_count * 2 <= len(valid):
        return reflection(prompt, _escalate=False)["answer"]

    for raw, norm in zip(raw_answers, normalized):
        if norm == winner:
            return raw
    return raw_answers[0]


def react(prompt, tools, max_steps=5):
    tool_names = ", ".join(sorted(tools.keys())) or "(none)"
    system = (
        f"Available tools: {tool_names}. On each step reply with one line: "
        f"'Action: <tool>[input]' or 'Final Answer: <answer>'."
    )
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
            try:
                obs = tools[name](arg)
            except Exception as e:
                obs = f"Error: {e}"
            history = history + "\nObservation: " + str(obs)

    final = call_model_chat_completions(
        history + "\nGive only the final answer.",
        system="Reply with only the final answer—no explanation."
    )
    return (final.get("text") or "").strip()


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


def reflection(question, max_retries=2, _escalate=True):
    attempt = cot(question)
    history = []
    converged = False

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
            converged = True
            break

        history.append(critique)
        retry_prompt = (
            f"{question}\n\n"
            "Previous attempts had these issues:\n- " + "\n- ".join(history) +
            "\n\nTry again, avoiding those issues."
        )
        attempt = cot(retry_prompt)

    #escalate to self_consistency if reflection does not converge on to a correct answer
    if _escalate and not converged:
        voted = self_consistency(question, n=3, _escalate=False)
        return {"answer": voted, "reasoning": attempt["reasoning"], "critiques": history}

    return {"answer": attempt["answer"], "reasoning": attempt["reasoning"], "critiques": history}


def process_json(input_path, output_path, workers: int = 4):
    with open(input_path, "r", encoding="utf-8") as fp:
        questions = json.load(fp)

    answers: list[dict | None] = [None] * len(questions)
    before = LLM_CALLS
    t0 = time.time()

    def work(i, q):
        try:
            return i, {"output": agent(q["input"])}
        except Exception as e:
            return i, {"output": None, "error": repr(e)}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, i, q) for i, q in enumerate(questions)]
        for done, fut in enumerate(as_completed(futures), 1):
            i, result = fut.result()
            answers[i] = result
            print(f"[{done}/{len(questions)}] done (idx {i})")

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)
    print(f"wrote {len(answers)} answers to {output_path}!")
    print(f"total llm calls: {LLM_CALLS - before} | wall time: {time.time() - t0:.1f}s")


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
    tools = tools or TOOLS
    tool_list = ", ".join(sorted(tools.keys()))
    system = f"Reason step by step. Available tools: {tool_list}. Call one with 'TOOL: <name>[input]'. When done write 'FINAL: answer'"
    history = prompt

    for _ in range(max_steps):
        result = call_model_chat_completions(history, system=system)
        text = (result.get("text") or "").strip()
        history = history + "\n" + text

        if "FINAL:" in text:
            return text.split("FINAL:")[-1].strip()

        match = re.search(r"TOOL:\s*(\w+)\[(.*?)\]", text, re.DOTALL)
        if match:
            name = match.group(1)
            arg = match.group(2)
            try:
                obs = tools[name](arg)
            except Exception as e:
                obs = f"Error: {e}"
            history = history + f"\nObservation: {obs}"

    #re ask without history for max tokens is not a problem
    final = call_model_chat_completions(
        f"Question: {prompt}\n\nGive ONLY the final answer.",
        system="Reply with only the final answer—no explanation."
    ).get("text") or ""
    return _strip_answer_markers(final)


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
    #pass to decomposition if no sequential sub problems
    if len(parts) < 2:
        return decomposition(question, max_parts=max_steps)

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
        "Classify the question into one label: "
        "compute (math, calculation, or code execution needed), "
        "decompose (problem has multiple steps or independent sub-questions), "
        "verify (factual or open-ended; correctness should be double-checked), "
        "simple (everything else). "
        "Reply with only the label."
    )
    result = call_model_chat_completions(question, system=system)
    label = (result.get("text") or "").strip().lower()
    valid = {"compute", "decompose", "verify", "simple"}
    if label not in valid:
        label = "simple"
    return label


def planning(question):
    system = (
        "You output a block-stacking plan in PDDL action syntax. "
        "One action per line, in parentheses, e.g. (unstack blue red), (stack blue yellow), "
        "(pick-up red), (put-down blue). Use lowercase action names (pick-up, put-down, stack, unstack) "
        "and lowercase block colors as arguments. "
        "Output only the parenthesized actions for the final [PLAN], no prose, no [PLAN]/[PLAN END] markers."
    )
    result = call_model_chat_completions(question, system=system)
    text = (result.get("text") or "").strip()
    actions = re.findall(r"([^)]*)", text)
    return "\n".join(actions)


def agent(question):
    if "[PLAN]" in question:
        return planning(question)
    label = classify(question)
    if label == "compute":
        return tool_augmented(question)
    if label == "decompose":
        return least_to_most(question)
    if label == "verify":
        return reflection(question)["answer"]
    return cot(question)["answer"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--out", default="answers.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.text is not None:
        before = LLM_CALLS
        print(agent(args.text))
        print(f"total llm calls: {LLM_CALLS - before}")
    else:
        process_json(args.json_path, args.out, workers=args.workers)


if __name__ == "__main__":
    main()
