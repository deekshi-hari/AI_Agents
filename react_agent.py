from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This functoion adds two numbers"""

    return a + b

@tool
def substract(a: int, b: int):
    """This functoion substract two numbers"""

    return a - b

@tool
def multiply(a: int, b: int):
    """multiplication"""

    return a * b

tools = [add, substract, multiply, ]

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content = "You are an assistant with tools: add, substract, multiply. "
                  "For ANY math operation, you MUST call the corresponding tool, "
                  "even if you can calculate it yourself. "
                  "If multiple steps are needed (e.g., add then multiply), "
                  "chain tool calls until you can return the final answer."
    )
    response = model.invoke([system_prompt] + state['messages'])

    return {'messages': [response]}


def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "substract 10 - 1 and multiply the result with 5 and then add the result with 100.")]}
print_stream(app.stream(inputs, stream_mode="values"))