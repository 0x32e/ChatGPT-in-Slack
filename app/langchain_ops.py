import time
import os
from typing import List, Dict

from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, Tool, AgentType, ZeroShotAgent, AgentExecutor, initialize_agent
from langchain.memory import PostgresChatMessageHistory, ConversationBufferMemory

from .langchain_tools import getLangchainTools

_llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

def execute_agent(
        thread_id: str,
        messages: List[Dict[str, str]], # message format: ["role": str, "content": str]
        user_id: str,
    ) -> str:
        print("Running agent...")

        last_user_input = _get_last_user_input(messages)

        message_history = PostgresChatMessageHistory(
             connection_string=f'{os.environ["POSTGRES_URI"]}/langchain', 
             session_id=thread_id,
             table_name="messages"
        )

        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)

        agent_chain = initialize_agent(
             tools=getLangchainTools(_llm), 
             llm=_llm, 
             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
             verbose=True, 
             memory=memory
        )
        return agent_chain.run(last_user_input)

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
