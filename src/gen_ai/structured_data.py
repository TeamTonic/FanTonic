from langchain.agents import AgentExecutor
from langchain.retrievers import PubMedRetriever
from langchain.retrievers import WikipediaRetriever
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import json
from langchain_community.utilities import GoogleSearchAPIWrapper

from langchain_core.agents import AgentActionMessageLog, AgentFinish
from typing import List, Dict

from langchain_core.pydantic_v1 import BaseModel, Field


# class Response(BaseModel):
#     """Final response to the question being asked"""
#     landmark: str = Field(description="The landmark of the given country")
#     # geolocation: str = Field(description="the x,y coordinates of the location") 
#     cultural_importance: str = Field(description="The cultural importance of the landmark") 
#     # cultural_importance: str = Field(description="") 

class Response(BaseModel):
    """Final response to the question being asked"""
    landmark: List[str] = Field(description="4 landmarks of a given country")
    # cultural_importance: str = Field(description="The cultural importance of the landmark") 

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )

llm = ChatOpenAI(temperature=1)

from langchain.tools.retriever import create_retriever_tool

# @tool
# def wikipedia_search(query: str) -> str:
#      """Search Wikipedia for additional information to expand on research papers or when no papers can be found."""
#     global all_sources

#     api_wrapper = WikipediaAPIWrapper()
#     wikipedia_search = WikipediaQueryRun(api_wrapper=api_wrapper)
#     wikipedia_results = wikipedia_search.run(query)
#     formatted_summaries = format_wiki_summaries(wikipedia_results)
#     all_sources += formatted_summaries
#     parsed_summaries = parse_list_to_dicts(formatted_summaries)
#     # add_many(parsed_summaries)
#     #all_sources += create_wikipedia_urls_from_text(wikipedia_results)
#     return wikipedia_results

retriever_tool = create_retriever_tool(
    # PubMedRetriever(),
    # GoogleSearchAPIWrapper(),
    WikipediaRetriever(),
    "biotech-research",
    "Query a retriever to get information about the genes that could solve the requested solution",
)

llm_with_tools = llm.bind_functions([retriever_tool, Response])

agent = (
    {
        "input": lambda x: x["input"],
        # Format agent scratchpad from intermediate steps
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | parse
)

agent_executor = AgentExecutor(tools=[retriever_tool], agent=agent, verbose=True)

if __name__ == "__main__":
    sample = agent_executor.invoke(
        # {"input": "I am looking for a gene that can be used for baker's yeast to prevent or even reconstitute the soil from drought. "},
        {"input": "find a landmark in Benin"},
        # return_only_outputs=True,
    )
    # x = 0
