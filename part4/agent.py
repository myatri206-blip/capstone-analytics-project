"""
Builds the single LangChain agent: a tool-calling agent + AgentExecutor,
with a bounded max_iterations so the reasoning loop can't run away.
"""
import os
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import get_weather, get_order_policy

load_dotenv()

TOOLS = [get_weather, get_order_policy]

SYSTEM_PROMPT = (
    "You are a helpful customer-support assistant for an online clothing "
    "retailer. You can check live weather (useful for questions about "
    "possible delivery delays) and look up official store policies. "
    "Use your tools whenever they would help answer the question — do "
    "not guess at information a tool could look up for you."
)


def build_agent_executor(max_iterations: int = 5, verbose: bool = True) -> AgentExecutor:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create a .env file with "
            "GEMINI_API_KEY=your_key_here (see README)."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)

    # max_iterations bounds the agent's reasoning loop so a confused model
    # can't call tools forever; return_intermediate_steps=True is what
    # lets us extract the real {tool, arguments} decision the framework
    # made, instead of hand-parsing raw text.
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
        verbose=verbose,
    )


def extract_tool_calls(intermediate_steps) -> list[dict]:
    """
    Extracts the {"tool": ..., "arguments": ...} decision the framework
    itself made for each tool call, from AgentExecutor's native
    intermediate_steps (a list of (AgentAction, observation) tuples).
    This is the framework's own structured record of the routing
    decision — not a hand-written regex over raw text.
    """
    calls = []
    for action, observation in intermediate_steps:
        calls.append({
            "tool": action.tool,
            "arguments": action.tool_input,
            "observation": str(observation),
        })
    return calls
