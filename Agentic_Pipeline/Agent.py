

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Union

from langchain_google_vertexai import ChatVertexAI
from google.oauth2 import service_account
import vertexai
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command

# 1. Authentication
credentials = service_account.Credentials.from_service_account_file(
    "/Users/yashkhatri/Downloads/stoked-producer-457308-m2-ddc5dd61eb91.json"
)
vertexai.init(project="stoked-producer-457308-m2", location="us-central1", credentials=credentials)

# 2. Define the LLM (Vertex AI Gemini)
llm = ChatVertexAI(
    model_name="gemini-2.0-flash-001",  # Fast Gemini model
    temperature=0.7,
)

planner_agent= create_react_agent(
    model=llm,
    tools=[DuckDuckGoSearchRun()],
    prompt="""
You are a planner agent. You will be given a user input. Your task is to create a plan for an itinerary based on the user's interest inputs. The user will provide you with the following variables:
1. Interest
2. Duration
3. Budget
"""
    
)
booking_agent= create_react_agent(
    model=llm,

    tools=[DuckDuckGoSearchRun()],
    prompt="""
You are a booking agent. You will be given a plan created by the planner agent. Your task is to get the best deals for the user based on the plan created by the planner agent."""
 
)

summarizer_agent= create_react_agent(
    model=llm,
    tools=[DuckDuckGoSearchRun()],
    prompt="""
You are a summarizer agent. You will be given a plan and bookings made by the planner and booking agents respectively. Your task is to summarize the plan and bookings in a concise manner."""
 
)

class State(MessagesState):
    next: str

member = {
    "planner": "The planner agent is responsible for creating a plan for an iternary based on the user's interest inputs.",
    "booking": "booking_agent is responsible for getting the best deals for the user based on the plan created by the planner agent.",
    "summarizer": "Based upon the plan created by the planner agent, the summarizer agent is responsible for creating a summary of the plan and required bookings.",

}
members= ["planner", "booking", "summarizer"]
options = members + ["FINISH"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {member}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]

print("This is a sample file.")

class State(MessagesState):
    next: str


def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})

def planner_node(state: State) -> Command[Literal["supervisor"]]:
    result = planner_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="planner")
            ]
        },
        goto="supervisor",
    )




def booking_node(state: State) -> Command[Literal["supervisor"]]:
    result = booking_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="booking_agent")
            ]
        },
        goto="supervisor",
    )

def summarizer_node(state: State) -> Command[Literal["supervisor"]]:
    result = summarizer_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="summarizer")
            ]
        },
        goto="supervisor",
    )



builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("planner", planner_node)
builder.add_node("booking", booking_node)
builder.add_node("summarizer", summarizer_node)
graph = builder.compile()
for s in graph.stream(
    {"messages": [("user", """I want to plan an itenary and my variables are:
                   1. Interest: Beach
                     2. Duration: 5 days
                        3. Budget: $1000
    """)]}, subgraphs=True
):
    print(s)
    print("----")

