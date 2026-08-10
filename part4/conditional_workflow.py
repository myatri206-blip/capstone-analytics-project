"""
A small conditional workflow, separate from the main agent loop, using
RunnablePassthrough.assign to accumulate state across chained steps, and
RunnableBranch to route to one of two different downstream chains based
on a condition evaluated over that accumulated state.

Flow:
  1. RunnablePassthrough.assign adds a "sentiment" key by running a
     classification chain over the incoming message.
  2. RunnableBranch inspects that "sentiment" value and routes to either
     the escalation-reply chain (negative sentiment) or the standard-
     reply chain (everything else).
"""
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def build_conditional_workflow():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. See README.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=api_key)

    # Step 1: classify sentiment
    classify_prompt = ChatPromptTemplate.from_template(
        "Classify the sentiment of this customer message as exactly one "
        "word: either 'negative' or 'other'. Message: {message}"
    )
    classify_chain = classify_prompt | llm | StrOutputParser()

    # Step 2a: escalation reply (used when sentiment == negative)
    escalation_prompt = ChatPromptTemplate.from_template(
        "Act as a customer service escalation specialist. The customer "
        "sent this message: \"{message}\". Write a short, urgent, "
        "genuinely apologetic reply that acknowledges the specific "
        "problem and offers to make it right immediately."
    )
    escalation_chain = escalation_prompt | llm | StrOutputParser()

    # Step 2b: standard reply (used otherwise)
    standard_prompt = ChatPromptTemplate.from_template(
        "Act as a friendly customer service representative. The customer "
        "sent this message: \"{message}\". Write a short, warm reply."
    )
    standard_chain = standard_prompt | llm | StrOutputParser()

    def is_negative(state: dict) -> bool:
        return "negative" in state["sentiment"].lower()

    full_workflow = (
        RunnablePassthrough.assign(sentiment=classify_chain)
        | RunnableBranch(
            (is_negative, escalation_chain),
            standard_chain,  # default branch
        )
    )
    return full_workflow, classify_chain
