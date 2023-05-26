import threading
import time
import re
from typing import List, Dict, Any, Generator, Tuple

import tiktoken

from slack_bolt import BoltContext
from slack_sdk.web import WebClient

from app.markdown import slack_to_markdown, markdown_to_slack
from app.slack_ops import update_wip_message

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

_llm = ChatOpenAI(temperature=0.0)

import os
if os.environ.get("SERPAPI_API_KEY") == "":
    raise Exception("Environment variable SERPAPI_API_KEY is missing")

_tools = load_tools(
    [
        "human",
        "serpapi",
        "llm-math",
    ]
)

_agent_chain = initialize_agent(
    _tools,
    _llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True,
)

default_template = """
I want you to act as a summarizer who summarizes the messages you received as input.
Below is the list of the messages in the ascending order of the time they were sent. 
Note that each message string takes the following format: "role: content", and the user your are talking to is the one with the role "user".
The user's user_id is {user_id}.

Messages:
{messages}
"""

def _convert_to_string(messages: List[Dict[str, str]]) -> str:
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

def execute_agent(
        messages: List[Dict[str, str]], # message format: ["role": str, "content": str]
        user_id: str,
    ) -> str:
        # TODO: Use LangChain's template to add a list of messages to the context
        prompt = PromptTemplate(
             input_variables=["messages", "user_id"],
             template=default_template,
        )
        return _agent_chain.run(prompt.format(messages=_convert_to_string(messages), user_id=user_id))