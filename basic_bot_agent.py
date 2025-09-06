import os
from dotenv import load_dotenv
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


def process(state: AgentState) -> AgentState:
    """This node will slove the request input of user"""
    print(f"sending to llm --------------------------------- {state["message"]}")
    response = llm.invoke(state["message"])

    state["message"].append(AIMessage(content=response.content))

    
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("You: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    res = agent.invoke(
                {"message": conversation_history}
                )
    conversation_history = res["message"]
    user_input = input("You: ")

    