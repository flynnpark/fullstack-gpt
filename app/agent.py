from typing import Type

import wikipedia
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pydantic.v1.types import SecretStr


class DuckDuckGoSearchToolArgsSchema(BaseModel):
    query: str = Field(
        title="Query",
        description="The search query to search for",
    )


class DuckDuckGoSearchTool(BaseTool):
    args_schema: Type[DuckDuckGoSearchToolArgsSchema] = DuckDuckGoSearchToolArgsSchema

    def __init__(self):
        super().__init__(
            name=self.__class__.__name__,
            description="Searches DuckDuckGo for the top search results",
        )

    def _run(self, query: str):
        duckduckgo = DuckDuckGoSearchAPIWrapper()
        return duckduckgo.run(query)


class Agent:
    def __init__(self, openai_key: SecretStr):
        llm = ChatOpenAI(temperature=0.1, api_key=openai_key)
        wiki_wrapper = WikipediaAPIWrapper(
            wiki_client=wikipedia, top_k_results=1, doc_content_chars_max=1000
        )
        prompt = hub.pull("hwchase17/openai-tools-agent")
        prompt.pretty_print()
        tools = [DuckDuckGoSearchTool(), WikipediaQueryRun(api_wrapper=wiki_wrapper)]
        agent = create_openai_tools_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
        )
        self.executor = AgentExecutor.from_agent_and_tools(agent, tools)

    def invoke(self, question: str):
        return self.executor.invoke({"input": question})
