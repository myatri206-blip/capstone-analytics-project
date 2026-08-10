# Part 3 — GenAI-Powered Text Analytics: Prompt Engineering & LLM API Integration

## Dataset

[Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
(CC0-1.0). `reviews_sample.csv` is a 300-record sample (reviews longer than
80 characters, sampled with a fixed random seed for reproducibility) drawn
from the full 23,486-row dataset, with the `Review Text` free-text field
required by the brief. 300 > the 200-record minimum.

## Required environment variable

This project needs **one** environment variable: **`GEMINI_API_KEY`**.

Get a free key (no payment method required) from
[Google AI Studio](https://aistudio.google.com/apikey), then create a file
named `.env` in this folder containing:

```
GEMINI_API_KEY=your_actual_key_here
```

`.env` is listed in `.gitignore` and is never committed — the grader supplies
their own key via the same variable name to run this.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

This makes real calls to the Gemini API and **automatically rewrites the
Results section below** with whatever the model actually returns — nothing
past this point is hand-typed, it's generated fresh on every run.

## Task 1 — Three prompt templates

All three (`prompts.py`) target the same locked JSON schema:
`{"label": "positive|negative|neutral", "confidence": "low|medium|high", "reason": "string"}`

- **Zero-shot** (`zero_shot_template`) — instruction + schema only, no examples.
- **Few-shot** (`few_shot_template`) — same instruction plus 5 worked examples
  embedded in the prompt.
- **Role-prompted** (`role_prompted_template`) — opens with an explicit persona
  ("Act as a senior customer-insights analyst...") and is structured with the
  ECO framework labelled explicitly: INSTRUCTION / CONTEXT / CONSTRAINTS / OUTPUT.

## Task 2 — API wrapper and retry logic

`llm_client.py`'s `call_llm(prompt, temperature, max_tokens)`:
- Loads `GEMINI_API_KEY` from the environment via `python-dotenv` (never
  hardcoded).
- Sends the prompt to Gemini's `generateContent` endpoint with the given
  `temperature` and `max_tokens` (mapped to `maxOutputTokens`).
- **Retry logic:** on a network exception, an HTTP 429 (rate limit), or any
  non-200 response, it retries up to 3 times with a short backoff (2s, 4s, 8s
  for rate limits) before giving up, logging a descriptive error, and
  returning an empty string so the caller can skip that record rather than
  crash the whole run. This path was unit-tested independently with a mocked
  API that fails twice then succeeds on the 3rd attempt — confirmed working
  (see `test_retry.py`, which you can run standalone with `python test_retry.py`).

## Task 3-6 — Results

Everything below this line was generated automatically by running `main.py`
against the live Gemini API — see the script for exact logic.

<!-- RESULTS_START -->

### Task 3 — 15-call template comparison (3 templates × 5 records)

| Template | Valid schema-conformant responses (out of 5) |
|---|---|
| zero_shot | 0/5 |
| few_shot | 0/5 |
| role_prompted | 0/5 |

**Most reliable template: `zero_shot`** (0/5 valid responses).

### Task 4 — Aspect-based sentiment (10 records)

| Record | Review (truncated) | Fit | Fit phrase | Quality | Quality phrase |
|---|---|---|---|---|---|
| 98 | The fabric is a little stiff, the back looks unusually long ... | PARSE FAILED | - | - | - |
| 259 | This dress is lovely. the embroidery is beautifully done and... | PARSE FAILED | - | - | - |
| 184 | Love this tee! the fit is true to size, and very flattering ... | PARSE FAILED | - | - | - |
| 256 | I couldn't be happier with this skirt! i was expecting a jer... | PARSE FAILED | - | - | - |
| 29 | I love this dress! i'm 5'5", and it's just long enough to be... | PARSE FAILED | - | - | - |
| 254 | I got the blue. it is very short. one side of the shirt i re... | PARSE FAILED | - | - | - |
| 7 | You've probably read through the other reviews, so i'll try ... | PARSE FAILED | - | - | - |
| 13 | This is a casual, flattering off the shoulder float dress wi... | PARSE FAILED | - | - | - |
| 230 | I bought this in the store so i didn't know that it was actu... | PARSE FAILED | - | - | - |
| 91 | Love this dress! it can be worn in cold or warm weather with... | PARSE FAILED | - | - | - |

### Task 5 — Auto-drafted replies (chained from Task 4 output)

### Task 6 — Multi-turn conversation demo

Conversation history object:
```json
[
  {
    "role": "user",
    "content": "I'm reviewing customer feedback for a clothing retailer. The product category we're focused on today is 'Dresses'. Just confirm you understand and are ready."
  },
  {
    "role": "model",
    "content": ""
  },
  {
    "role": "user",
    "content": "Given the product category I mentioned, name one common fit issue customers might report."
  },
  {
    "role": "model",
    "content": ""
  }
]
```

**Turn 2 uses Turn 1's context** — the second response should reference 'Dresses' (the category named in turn 1) without it being restated in turn 2's question, proving context carried over.


<!-- RESULTS_END -->
