from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

class AgentState(TypedDict):
    message: List[HumanMessage]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["message"])
    print(f"\nAI: {response.content}")


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("You: ")

while user_input != "exit":
    agent.invoke(
        {"message": [HumanMessage(content=user_input)]}
        )
    user_input = input("You: ")