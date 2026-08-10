"""
Part 3 — main script. Runs, in order:
  Task 3: 15-call comparison (3 templates x 5 records), JSON-parse each
  Task 4: aspect-based sentiment on 10 records using the best template
  Task 5: chained response-drafting for 3+ of those records
  Task 6: a 2-turn multi-turn conversation demo

All real results are written directly into README.md, between the
<!-- RESULTS_START --> / <!-- RESULTS_END --> markers, by this script.
Nothing in README.md's results section is hand-typed — it's generated
from whatever your own API key actually returns when you run this.

Run: python main.py
"""
import json
import pandas as pd

from llm_client import call_llm, parse_json_response
from prompts import zero_shot_template, few_shot_template, role_prompted_template

df = pd.read_csv("reviews_sample.csv", index_col=0)

TEMPLATES = {
    "zero_shot": zero_shot_template,
    "few_shot": few_shot_template,
    "role_prompted": role_prompted_template,
}

output_lines = []  # collects markdown to inject into README.md


def log(md_line=""):
    print(md_line)
    output_lines.append(md_line)


# ================================================================
# Task 3: run all three templates on the same 5 records (15 calls)
# ================================================================
log("### Task 3 — 15-call template comparison (3 templates × 5 records)\n")

five_records = df.sample(n=5, random_state=1)
valid_counts = {name: 0 for name in TEMPLATES}
comparison_rows = []

for record_id, row in five_records.iterrows():
    review = row["Review Text"]
    for template_name, template_fn in TEMPLATES.items():
        prompt = template_fn(review)
        raw = call_llm(prompt, temperature=0.2, max_tokens=200)
        parsed = parse_json_response(raw)
        required_fields = {"label", "confidence", "reason"}
        is_valid = parsed is not None and required_fields.issubset(parsed.keys())
        if is_valid:
            valid_counts[template_name] += 1
        else:
            print(f"  LOG: record={record_id} template={template_name} "
                  f"FAILED to parse valid schema. Raw response: {raw[:150]!r}")
        comparison_rows.append({
            "record_id": record_id,
            "template": template_name,
            "valid_json": is_valid,
            "label": parsed.get("label") if parsed else None,
        })

best_template = max(valid_counts, key=valid_counts.get)

log("| Template | Valid schema-conformant responses (out of 5) |")
log("|---|---|")
for name, count in valid_counts.items():
    log(f"| {name} | {count}/5 |")
log(f"\n**Most reliable template: `{best_template}`** "
    f"({valid_counts[best_template]}/5 valid responses).\n")

# ================================================================
# Task 4: aspect-based sentiment on 10 records, using the best template
# ================================================================
log("### Task 4 — Aspect-based sentiment (10 records)\n")

ten_records = df.sample(n=10, random_state=2)

ASPECT_PROMPT_TEMPLATE = """Act as a senior customer-insights analyst.
Analyze the following clothing review for TWO specific aspects: "fit" and
"quality". For each aspect, provide a sentiment label and a short
actionable phrase (3-6 words) describing what was liked or disliked
about that aspect specifically. If an aspect isn't mentioned at all,
label it "not_mentioned" and give the actionable phrase "no comment on this aspect".

Respond with ONLY a JSON object, no markdown fences, in exactly this schema:
{
  "fit": {"label": "positive|negative|neutral|not_mentioned", "actionable_phrase": "string"},
  "quality": {"label": "positive|negative|neutral|not_mentioned", "actionable_phrase": "string"}
}

Review: "{review}"
"""

aspect_results = []
for record_id, row in ten_records.iterrows():
    review = row["Review Text"]
    prompt = ASPECT_PROMPT_TEMPLATE.replace("{review}", review)
    raw = call_llm(prompt, temperature=0.2, max_tokens=300)
    parsed = parse_json_response(raw)
    aspect_results.append({"record_id": record_id, "review": review, "parsed": parsed})

log("| Record | Review (truncated) | Fit | Fit phrase | Quality | Quality phrase |")
log("|---|---|---|---|---|---|")
for r in aspect_results:
    review_short = r["review"][:60].replace("|", "/") + "..."
    if r["parsed"]:
        fit = r["parsed"].get("fit", {})
        quality = r["parsed"].get("quality", {})
        log(f"| {r['record_id']} | {review_short} | {fit.get('label')} | "
            f"{fit.get('actionable_phrase')} | {quality.get('label')} | "
            f"{quality.get('actionable_phrase')} |")
    else:
        log(f"| {r['record_id']} | {review_short} | PARSE FAILED | - | - | - |")
log()

# ================================================================
# Task 5: chain into a response-drafting prompt for 3+ records
# ================================================================
log("### Task 5 — Auto-drafted replies (chained from Task 4 output)\n")

DRAFT_PROMPT_TEMPLATE = """Act as a customer service representative for a
women's clothing retailer. A customer left this review:

"{review}"

Our analysis found: fit sentiment = {fit_label} ({fit_phrase}); quality
sentiment = {quality_label} ({quality_phrase}).

Write a short (3-4 sentence), professional, empathetic reply that
specifically addresses the fit and quality points raised — do not write a
generic "thank you for your feedback" reply. Reference the actual issue(s)
or praise the customer mentioned.
"""

drafted = 0
for r in aspect_results:
    if drafted >= 3:
        break
    if not r["parsed"]:
        continue
    fit = r["parsed"].get("fit", {})
    quality = r["parsed"].get("quality", {})
    prompt = DRAFT_PROMPT_TEMPLATE.format(
        review=r["review"],
        fit_label=fit.get("label"), fit_phrase=fit.get("actionable_phrase"),
        quality_label=quality.get("label"), quality_phrase=quality.get("actionable_phrase"),
    )
    reply = call_llm(prompt, temperature=0.4, max_tokens=200)
    log(f"**Record {r['record_id']}**")
    log(f"> Original review: \"{r['review'][:200]}...\"")
    log(f">")
    log(f"> Drafted reply: {reply.strip()}")
    log()
    drafted += 1

# ================================================================
# Task 6: multi-turn context demonstration
# ================================================================
log("### Task 6 — Multi-turn conversation demo\n")

turn1_prompt = ("I'm reviewing customer feedback for a clothing retailer. "
                 "The product category we're focused on today is 'Dresses'. "
                 "Just confirm you understand and are ready.")
turn1_response = call_llm(turn1_prompt, temperature=0.2, max_tokens=100)

conversation_history = [
    {"role": "user", "content": turn1_prompt},
    {"role": "model", "content": turn1_response},
]

turn2_prompt = "Given the product category I mentioned, name one common fit issue customers might report."
# Build a single prompt that includes the prior turn's context, since the
# Gemini REST call in llm_client.call_llm() is single-turn; we manually
# thread the history into the prompt to demonstrate context carry-over.
turn2_full_prompt = (
    f"Earlier in this conversation:\nUser: {turn1_prompt}\nAssistant: {turn1_response}\n\n"
    f"Now: {turn2_prompt}"
)
turn2_response = call_llm(turn2_full_prompt, temperature=0.2, max_tokens=150)
conversation_history.append({"role": "user", "content": turn2_prompt})
conversation_history.append({"role": "model", "content": turn2_response})

log("Conversation history object:")
log("```json")
log(json.dumps(conversation_history, indent=2))
log("```")
log()
log(f"**Turn 2 uses Turn 1's context** — the second response should reference "
    f"'Dresses' (the category named in turn 1) without it being restated in "
    f"turn 2's question, proving context carried over.\n")

# ================================================================
# Write everything into README.md between the markers
# ================================================================
results_markdown = "\n".join(output_lines)

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

start_marker = "<!-- RESULTS_START -->"
end_marker = "<!-- RESULTS_END -->"
before = readme.split(start_marker)[0]
after = readme.split(end_marker)[1]
new_readme = before + start_marker + "\n\n" + results_markdown + "\n\n" + end_marker + after

with open("README.md", "w", encoding="utf-8") as f:
    f.write(new_readme)

print("\n\nDone. README.md has been updated with real results from this run.")
