"""
Copyright 2026 Luca Silver

Root routing agent that directs requests to either student or advisor chat graphs based on account type.

Functions:
- `route`: Conditional router that directs to student or advisor graph based on account_type.
- `invoke_s_graph`: Looks up student ID from parent user ID and invokes the student chat graph.
- `invoke_a_graph`: Invokes the advisor chat graph with the current conversation state.

Graph Structure:
- START -> [invoke_s_graph | invoke_a_graph] (conditional routing based on account type)
- invoke_s_graph -> END
- invoke_a_graph -> END

Exports:
- `chat_graph`: Compiled LangGraph root routing agent.
"""

import sys, os

# adds lg_agent directory to system path if not already there
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from lg_agent.utilities.state import RouteState
from s_chat_graph import s_chat_graph
from a_chat_graph import a_chat_graph
from data_pipeline.database.database_dev_tools import __connect, aconnect
from typing import Literal


# cd my-agent && .venv\Scripts\activate && npx @langchain/langgraph-cli dev --port 8123 --no-browser
load_dotenv()

def route(state: RouteState) -> Literal["invoke_s_graph", "invoke_a_graph"]:
    """
    Chooses the main chat graph based on the user's account type.

    Student users are routed to the student chat graph, while advisor users are
    routed to the advisor chat graph.

    Args:
        state (RouteState): Routing state containing the authenticated user ID,
            account type, and current messages.

    Returns:
        Literal["invoke_s_graph", "invoke_a_graph"]: The next node name for the
            state graph.
    """
    account_type = state.get("account_type")
    
    match account_type:
        case "Student":
            return "invoke_s_graph"
        case "Advisor":
            return "invoke_a_graph"
        case _:
            raise ValueError(f"Invalid account type: {account_type}. Must be 'Student' or 'Advisor'.")

async def invoke_s_graph(state: RouteState) -> RouteState:
    """
    Resolves the current student ID and invokes the student chat graph.

    The route state contains the user ID. This node looks up the corresponding student ID, 
    then seeds the student chat graph with the message history and the student-specific 
    identifiers needed downstream.

    Args:
        state (RouteState): Routing state containing the current messages, user ID,
            and account type.

    Returns:
        RouteState: State update containing the messages returned by the student graph.

    Raises:
        ValueError: If no student row exists for the provided parent user ID.
    """
    async with aconnect() as conn:
        print("CHAT AGENT RECEIVED STATE:", state, flush=True)
        cur = await conn.execute("SELECT ID FROM students WHERE ParentID = ?", (state["user_id"],))
        student_id = await cur.fetchone()
        if student_id is None:
            raise ValueError(f"No student found for parent_id {state['user_id']}")
        result = await s_chat_graph.ainvoke({"messages": state["messages"], "plan": {}, "loop_count": 0, "user_id": student_id[0], "insertion_result": ""})
    return {"messages": result["messages"]}

async def invoke_a_graph(state: RouteState) -> RouteState:
    """
    Invokes the advisor chat graph with the current conversation state.

    Args:
        state (RouteState): Routing state containing the current messages, user ID,
            and account type.

    Returns:
        RouteState: State update containing the messages returned by the advisor graph.
    """
    result = await a_chat_graph.ainvoke({"messages": state["messages"], "plan": {}, "loop_count": 0, "user_id": state["user_id"]})
    return {"messages": result["messages"]}

graph_builder = StateGraph(RouteState)

graph_builder.add_node("invoke_s_graph", invoke_s_graph)
graph_builder.add_node("invoke_a_graph", invoke_a_graph)

graph_builder.add_conditional_edges(START, route, ["invoke_s_graph", "invoke_a_graph"])
graph_builder.add_edge("invoke_s_graph", END)
graph_builder.add_edge("invoke_a_graph", END)

chat_graph = graph_builder.compile()