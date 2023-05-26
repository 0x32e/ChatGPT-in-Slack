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
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory

from langchain.tools import StructuredTool
from llama_index import GPTVectorStoreIndex, download_loader, GPTVectorStoreIndex

_llm = ChatOpenAI(temperature=0.0)

import os

if os.environ.get("GPLACES_API_KEY") == "":
    raise Exception("Environment variable GPLACES_API_KEY is missing")

if os.environ.get("SERPAPI_API_KEY") == "":
    raise Exception("Environment variable SERPAPI_API_KEY is missing")

# UnstructuredURLLoader = download_loader("UnstructuredURLLoader")

# TODO: NOT WORKING...
# Does summarize the content, but it seems that the response is not being sent back to the user.
# @tool("summarize_webpage", return_direct=True)
# def summarizeWebpage(url: str) -> str:
#     """Load a webpage with the given url and return the summary of the content. Return the summary to the user immediately."""
#     loader = UnstructuredURLLoader(urls=[url], continue_on_failure=False, headers={})
#     docs = loader.load()
#     index = GPTVectorStoreIndex.from_documents(docs)
#     return index.as_query_engine().query("Return summary")

_tools = load_tools(
    [
        "llm-math",
        "serpapi",
    ],
    llm=_llm
)

# _tools.append(summarizeWebpage)

_memory = ConversationBufferMemory(memory_key="chat_history")

_agent_chain = initialize_agent(
    tools=_tools,
    llm=_llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    memory=_memory,
)

# default_template = """
# I want you to act as a summarizer who summarizes the messages you received as input. Below is the list of the messages in the ascending order of the time they were sent. 
# Note that each message string takes the following format: "role: content", and the user your are talking to is the one with the role "user" with the user_id of {user_id}.
# Output should be a simple one or two sentence summary of the conversations between you and the user.

# Messages:
# {messages}
# """

default_template = """{user_input}"""

# def _convert_to_user_input_and_context(messages: List[Dict[str, str]]):
#     last_user_input = ""
#     last_user_input_index = -1
#     for i in range(len(messages)-1, 0, -1):
#          if messages[i].get("role") == "user":
#               last_user_input_index = i
#               last_user_input = messages[i].get("content")
#               break
#     if last_user_input_index == -1:
#         raise Exception("No user input found")
    
#     # context = "\n".join([f"{m['role']}: {m['content']}" for i, m in messages if i != last_user_input_index])
#     context = ""
#     for i in range(len(messages)):
#          if i != last_user_input_index:
#               m = f"{messages[i]['role']}: {messages[i]['content']}"
#               context += "\n" + m
#     return last_user_input, context

def _get_last_user_input(messages: List[str]) -> str:
    last_user_input = ""
    last_user_input_index = -1
    for i in range(len(messages)-1, 0, -1):
         if messages[i].get("role") == "user":
              last_user_input_index = i
              last_user_input = messages[i].get("content")
              break
    if last_user_input_index == -1:
        raise Exception("No user input found")
    return last_user_input

def execute_agent(
        messages: List[Dict[str, str]], # message format: ["role": str, "content": str]
        user_id: str,
    ) -> str:
        # Use LangChain's template to add a list of messages to the context
        print("Running agent...")
        prompt = PromptTemplate(
             input_variables=["user_input"],
             template=default_template,
        )
        last_user_input = _get_last_user_input(messages)
        formattedPrompt = prompt.format(user_input=last_user_input)
        return _agent_chain.run(input=formattedPrompt)