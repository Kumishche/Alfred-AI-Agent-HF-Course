from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

import os
from dotenv import load_dotenv

load_dotenv()


from tools import weather_info_tool, hub_stats_tool
from retriever import guest_info_tool, draft_conversation



search_tool = DuckDuckGoSearchRun()

llm = ChatOpenAI(
    model=os.getenv('LLM_MODEL'),
    base_url=os.getenv('BASE_URL'),
    api_key=os.getenv('API_KEY')
)

tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool, draft_conversation]
chat_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        'messages': [chat_with_tools.invoke([
            SystemMessage(
                content="""You are Alfred, a helpful personal assistant."""
                ), 
             *state['messages'],
            ])
        ],
    }

builder = StateGraph(AgentState)

builder.add_node('assistant', assistant)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, 'assistant')
builder.add_conditional_edges(
    'assistant',
    tools_condition,
)
builder.add_edge('tools', 'assistant')

alfred = builder.compile()


while True:
    print('Ask Alfred a question: ')
    # response = alfred.invoke({'messages': [HumanMessage(content=input())]})
    for step in alfred.stream({'messages': [HumanMessage(content=input())]}, stream_mode='updates'):
        for node, update in step.items():
            for msg in update['messages']:
                msg.pretty_print()
    print('-'*20)
    print('Alfred\'s response:' )
    # print(response['messages'][-1].content)
    print('='*40)
    print()
    