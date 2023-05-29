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
from langchain.agents import Tool
from langchain.memory import ConversationStringBufferMemory

from langchain.tools import StructuredTool
from llama_index import GPTVectorStoreIndex, download_loader, GPTVectorStoreIndex

from langchain.utilities import PythonREPL
from langchain.utilities import BingSearchAPIWrapper

import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import DuckDuckGoSearchRun

import os

# if os.environ.get("BING_SEARCH_URL") == "":
#     raise Exception("Environment variable BING_SEARCH_URL is missing") 

# if os.environ.get("BING_SUBSCRIPTION_KEY") == "":
#     raise Exception("Environment variable BING_SUBSCRIPTION_KEY is missing") 

embeddings_model = OpenAIEmbeddings()
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

_llm = ChatOpenAI(temperature=0.0)
_python_repl = PythonREPL()

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
    ],
    llm=_llm
)

# You can create the tool to pass to an agent
# repl_tool = Tool(
#     name="python_repl",
#     description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`. You can also use this if the other Calculator cannot execute the operation.",
#     func=_python_repl.run
# )

# _tools.append(repl_tool)

# _bingSearch = BingSearchAPIWrapper()
# _bingTool = Tool(
#      name="bing_search",
#      description="A Bing search engine. Use this to search for information on the web. Input should be a valid search query.",
#      func=_bingSearch.run
# )

# _tools.append(_bingTool)

_web_search = DuckDuckGoSearchRun()
_tools.append(_web_search)

_memory = ConversationStringBufferMemory(memory_key="chat_history")

_agent_chain = initialize_agent(
    tools=_tools,
    llm=_llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    memory=_memory,
    max_iterations=10,
)

default_template = """[SYSTEM]:
You are a helpful conversational agent who helps the user with their questions. 
Current date is {current_time}.

[USER]:
{user_input}

[CONTEXT]:
- You have access to the following tools: {tools}
- You are talking to the user with the user_id of {user_id}
- The user is based in New York City
- When you are asked about the current time, use the current date above.
"""

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

def _get_tool_names() -> str:
     tools = [f"Name: {tool.name}, description: {tool.description}" for tool in _tools]
     return "\n".join(tools)

def _get_current_time() -> str:
     # set the timezone offset for ET
    tz_offset = -4 * 60 * 60

    # get the current time in seconds since the Epoch
    current_time = time.time()

    # add the timezone offset to the current time
    et_time = current_time + tz_offset

    # convert the Unix timestamp to a datetime object
    et_datetime = time.strftime('%m/%d/%Y, %I:%M:%S %p', time.localtime(et_time))

    # append the timezone abbreviation
    et_datetime += ' ET'

    return et_datetime

def execute_agent(
        messages: List[Dict[str, str]], # message format: ["role": str, "content": str]
        user_id: str,
    ) -> str:
        # Use LangChain's template to add a list of messages to the context
        print("Running agent...")
        prompt = PromptTemplate(
             input_variables=["user_input", "current_time", "tools", "user_id"],
             template=default_template,
        )
        last_user_input = _get_last_user_input(messages)
        formattedPrompt = prompt.format(
             user_input=last_user_input, 
             current_time=_get_current_time(),
             tools=_get_tool_names(),
             user_id=user_id,
        )
        return _agent_chain.run(input=formattedPrompt)