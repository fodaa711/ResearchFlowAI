import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.tools.tools import web_search, scrape_url

load_dotenv()


# -----------------------------
# Model Initialization
# -----------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


# -----------------------------
# Search Agent
# -----------------------------
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
    )


# -----------------------------
# Reader Agent
# -----------------------------
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
    )


# -----------------------------
# Writer Chain
# -----------------------------
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research writer. Write clear, structured and insightful reports.",
        ),
        (
            "human",
            """Write a detailed research report on the topic below.

Topic:
{topic}

Research Gathered:
{research}

Structure the report as:

# Introduction

# Key Findings
(At least 3 detailed points)

# Conclusion

# Sources
(List every URL mentioned in the research)

The report should be professional, factual, and well organized.
""",
        ),
    ]
)

writer_chain = writer_prompt | llm | StrOutputParser()


# -----------------------------
# Critic Chain
# -----------------------------
critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research reviewer. Review reports critically and provide constructive feedback.",
        ),
        (
            "human",
            """Review the report below.

Report:
{report}

Respond in this format:

Score: X/10

Strengths:
- ...
- ...

Weaknesses:
- ...
- ...

Suggestions:
- ...
- ...

Final Verdict:
...
""",
        ),
    ]
)

critic_chain = critic_prompt | llm | StrOutputParser()
