import os
from typing import List

from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool, load_tools
from langchain.base_language import BaseLanguageModel
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.tools import BaseTool

def _getGoogleSerperTool() -> Tool:
    return Tool(
        name="Search",
        func=GoogleSerperAPIWrapper().run,
        description="A search engine. Useful for when you need to answer questions about current events. Input should be a search query."
    )

def _dbTool(llm: BaseLanguageModel) -> List[BaseTool]:
    db = SQLDatabase.from_uri(os.environ["POSTGRES_URI"])
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()

def getLangchainTools(llm: BaseLanguageModel) -> List[BaseTool]:
    tls = load_tools(
        [
            "llm-math",
        ],
        llm=llm
    )

    for t in _dbTool(llm):
        tls.append(t)

    tls.append(_getGoogleSerperTool())

    return tls
