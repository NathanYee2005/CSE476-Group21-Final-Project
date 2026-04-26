# CSE 476 Final Project

An inference-time agent for solving reasoning requests by routing the questions through the following reasoning strategies: Chain-of-Thought, ReAct, Self-Consistency, Reflection, Least-to-Most, Decomposition, Self-Refine, and Planning

---

## Prerequisites

- Python 3.10 or higher (the code uses `str | None` union syntax)
- An OpenAI API key

---

## Installation

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd <repo-folder>
```

**2. Create and activate a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` and set your values:

```env
OPENAI_API_KEY=your_voyager_portal_api_key_here
API_BASE=https://openai.rc.asu.edu/v1
MODEL_NAME=qwen3-30b-a3b-instruct-2507
```

---

## Usage

### Process a JSON dataset (default mode)

Runs the agent over all questions in the provided JSON file and writes results to an output file. This argument is required.

```bash
python starting.py data/cse476_final_project_dev_data.json
```

Optionally specify a custom output file (default is answers.json) and number of parallel workers (default is 4):

```bash
python starting.py data/cse476_final_project_dev_data.json --out out.json --workers 8
```

### Ask a single question

Pass a question directly on the command line for a quick test:

```bash
python starting.py --text "Which musical featured the song Flash Bang, Wallop?"
```

---

## Input / Output Format

**Input JSON** — a list of objects with an `"input"` field:

```json
[
    {
        "input": "What is the product of the real roots of the equation $x^2 + 18x + 30 = 2 \\sqrt{x^2 + 18x + 45}$ ?"
    },
    {
        "input": "Which musical featured the song Flash Bang, Wallop?"
    }
]
```

**Answers JSON** — a list of objects with an `"output"` field:

```json
[
    { 
        "output": "20" 
    },
    {
        "output": "half"
    }
]
```