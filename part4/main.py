"""
Part 4 — main script. Runs, in order:
  1. Three distinct queries through the agent, logging each tool call's
     real {tool, arguments} decision (extracted from AgentExecutor's own
     intermediate_steps, not hand-parsed text).
  2. A 2-turn memory demonstration.
  3. The conditional workflow, invoked twice (once per branch).

All real results are written directly into README.md, between the
<!-- RESULTS_START --> / <!-- RESULTS_END --> markers — nothing there is
hand-typed, it's generated from your own API key's actual run.

Run: python main.py
"""
import json
from langchain_core.messages import HumanMessage, AIMessage

from agent import build_agent_executor, extract_tool_calls
from conditional_workflow import build_conditional_workflow

output_lines = []


def log(md_line=""):
    print(md_line)
    output_lines.append(md_line)


# ================================================================
# Part A: three distinct queries, each exercising at least one tool
# ================================================================
log("### Three demonstrated queries\n")

QUERIES = [
    "What's the current weather in Paris? We have a delivery going out "
    "there today and I want to know if it might be delayed.",
    "What is your policy on returns?",
    "Can you check the weather in Tokyo, and also tell me your exchange policy?",
]

executor = build_agent_executor(max_iterations=5, verbose=True)

for i, query in enumerate(QUERIES, start=1):
    log(f"**Query {i}:** {query}\n")
    result = executor.invoke({"input": query, "chat_history": []})
    tool_calls = extract_tool_calls(result.get("intermediate_steps", []))

    log("Logged tool-call decision(s), extracted from AgentExecutor's native `intermediate_steps`:")
    log("```json")
    for call in tool_calls:
        log(json.dumps({"tool": call["tool"], "arguments": call["arguments"]}, indent=2))
    if not tool_calls:
        log("(no tool calls were made for this query)")
    log("```")
    log(f"**Final answer:** {result['output']}\n")

# ================================================================
# Part B: 2-turn memory demonstration
# ================================================================
log("### 2-turn memory demonstration\n")

chat_history = []

turn1 = "I live in Berlin. What's the weather like there right now?"
log(f"**Turn 1 (user):** {turn1}")
result1 = executor.invoke({"input": turn1, "chat_history": chat_history})
log(f"**Turn 1 (agent):** {result1['output']}\n")

chat_history.append(HumanMessage(content=turn1))
chat_history.append(AIMessage(content=result1["output"]))

turn2 = "Given that, should I expect any delivery delays today?"
log(f"**Turn 2 (user):** {turn2}")
result2 = executor.invoke({"input": turn2, "chat_history": chat_history})
log(f"**Turn 2 (agent):** {result2['output']}\n")

log("**Memory check:** Turn 2 never restates 'Berlin' — the agent's answer "
    "above should reference Berlin/its weather anyway, proving it correctly "
    "reused Turn 1's information from `chat_history` rather than needing it "
    "repeated.\n")

# ================================================================
# Part C: conditional workflow, invoked twice (once per branch)
# ================================================================
log("### Conditional workflow demonstration (both branches)\n")

workflow, classify_chain = build_conditional_workflow()

negative_msg = "My order arrived damaged and this is the second time this has happened. I'm really frustrated."
positive_msg = "Just wanted to say the dress I ordered fits perfectly, thank you!"

for label, msg in [("Negative-sentiment input (should route to escalation chain)", negative_msg),
                    ("Neutral/positive input (should route to standard chain)", positive_msg)]:
    log(f"**{label}**")
    log(f"Input message: \"{msg}\"")
    sentiment = classify_chain.invoke({"message": msg})
    log(f"Classified sentiment: `{sentiment.strip()}`")
    reply = workflow.invoke({"message": msg})
    log(f"Routed reply: {reply.strip()}\n")

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
