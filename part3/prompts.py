"""
Task 1 — Three prompt templates for sentiment classification on customer
clothing reviews. All three lock the model into the SAME JSON schema:

{
  "label": "positive|negative|neutral",
  "confidence": "low|medium|high",
  "reason": "string"
}

so that responses across templates are directly comparable.
"""

JSON_SCHEMA_INSTRUCTION = """
Respond with ONLY a single JSON object, no markdown code fences, no extra text
before or after it, matching exactly this schema:
{
  "label": "positive|negative|neutral",
  "confidence": "low|medium|high",
  "reason": "a short string explaining why"
}
"""


def zero_shot_template(review_text: str) -> str:
    """(a) Zero-shot: instruction only, no worked examples."""
    return f"""Classify the sentiment of the following customer clothing review
as positive, negative, or neutral.
{JSON_SCHEMA_INSTRUCTION}
Review: "{review_text}"
"""


def few_shot_template(review_text: str) -> str:
    """(b) Few-shot: same instruction plus worked examples embedded in the prompt."""
    examples = """
Examples:

Review: "This dress fits perfectly and the fabric feels amazing. I get compliments every time I wear it."
Output: {"label": "positive", "confidence": "high", "reason": "Explicit praise for fit and fabric quality"}

Review: "The stitching came undone after one wash. Very disappointed with the quality."
Output: {"label": "negative", "confidence": "high", "reason": "Product failed after minimal use, explicit disappointment"}

Review: "It's an okay top. Nothing special but nothing wrong with it either."
Output: {"label": "neutral", "confidence": "medium", "reason": "Explicitly indifferent, no strong positive or negative signal"}

Review: "Runs a bit small, so I sized up. Once I did, it was great!"
Output: {"label": "positive", "confidence": "medium", "reason": "Initial sizing issue resolved, ends on a positive note"}

Review: "The color in person is nothing like the photo. Returning it."
Output: {"label": "negative", "confidence": "high", "reason": "Product misrepresented, customer is returning it"}
"""
    return f"""Classify the sentiment of a customer clothing review as
positive, negative, or neutral. Use the pattern shown in the examples below.
{examples}
{JSON_SCHEMA_INSTRUCTION}
Now classify this review:
Review: "{review_text}"
"""


def role_prompted_template(review_text: str) -> str:
    """
    (c) Role-prompted: explicit persona line, structured with the ECO
    framework (Instruction, Context, Constraints, Output) explicitly labelled.
    """
    return f"""Act as a senior customer-insights analyst at a women's clothing
retailer with 10 years of experience reading customer feedback at scale.

INSTRUCTION:
Classify the sentiment of the customer review provided below.

CONTEXT:
You work for an online clothing retailer. Reviews come from verified
purchasers and may discuss fit, fabric quality, color accuracy, comfort,
or overall satisfaction. Your classification feeds directly into a
weekly product-quality dashboard used by the merchandising team, so
accuracy and calibrated confidence matter.

CONSTRAINTS:
- Base your label ONLY on the review text given, not on assumptions about
  the product.
- If the review expresses mixed sentiment, weigh the overall tone the
  customer leaves the reader with.
- Do not invent details not present in the review.

OUTPUT:
{JSON_SCHEMA_INSTRUCTION}

Review: "{review_text}"
"""
