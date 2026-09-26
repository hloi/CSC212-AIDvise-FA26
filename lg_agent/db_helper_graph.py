"""
Copyright 2026 Luca Silver

Database helper subgraph that executes database queries for the planning agents.

Functions:
- `format_db_output`: Normalizes database helper response into standardized helper output schema.
- `tool_route`: Conditionally routes database helper loop based on tool calls and loop count.

Graph Structure:
- START -> db_node
- db_node -> [tool_node | alt_tool_node | format_db_output] (conditional routing based on tool_route)
- tool_node -> db_node (feedback loop)
- alt_tool_node -> db_node (feedback loop)
- format_db_output -> END

Exports:
- `db_graph`: Compiled LangGraph database helper subgraph.

Supports both standard (student) and alternate (advisor) tool sets.
Loop limits are defined in config.json.
"""

import sys, os

# adds lg_agent directory to system path if not already there
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from utilities.state import DatabaseHelperState, DatabaseHelperOutput
from utilities.nodes import db_node
from utilities.tools import db_tools, alt_db_tools
import json

CONFIG_PATH = os.path.join(PARENT_DIR, "config.json")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)
    LOOP_CONFIG = CONFIG["loop_limits"]

tool_node = ToolNode(db_tools)
alt_tool_node = ToolNode(alt_db_tools)

def format_db_output(state: DatabaseHelperState) -> DatabaseHelperOutput:
    """
    Normalizes the database helper response into the helper output schema.

    If the last message is a tool message (meaning the database helper passed its 
    loop limit without providing a final answer), the full message history is serialized
    into a single string so the planner can inspect the tool trail. Otherwise, the
    final AI message content is returned directly as the query result.

    Args:
        state (DatabaseHelperState): Helper state containing the request context and
            the message history produced by the database helper loop.

    Returns:
        DatabaseHelperOutput: Output object containing the original query and the
            formatted database result string.
    """
    if isinstance(state["messages"][-1], ToolMessage):
        message_dump = ""
        for message in state["messages"]:
            message_str = f"{message.role}: {message.content}\n\n"
            message_dump += message_str
        result = message_dump
    else:
        result = state["messages"][-1].content
    print(f"DB HELPER COMPLETE result_chars={len(str(result))}", flush=True)
    return {"info": {"query": state["info_needed"], "result": result}}

def tool_route(state: DatabaseHelperState):
    """
    Routes the database helper loop based on tool usage and loop count.

    The database helper is allowed a limited number of iterations. If the latest AI
    message includes tool calls, the graph routes to the appropriate tool node.
    Otherwise it formats the output. If the loop limit is reached, the function
    enforces a final response path and includes a fallback message when necessary.

    Args:
        state (DatabaseHelperState): Current helper state including messages,
            account type, loop counter, and requested information.

    Returns:
        str: The next node name to execute.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if state["loop_count"] < LOOP_CONFIG["s-db"] if state["account_type"] == "Student" else LOOP_CONFIG["a-db"]:
        if getattr(last_message, "tool_calls", None):
            tool_names = [
                tool_call.get("name", "unknown")
                for tool_call in last_message.tool_calls
            ]
            if state["account_type"] == "Student":
                print(
                    f"DB HELPER ROUTE destination=tool_node "
                    f"loop={state['loop_count']} tool_calls={tool_names}",
                    flush=True,
                )
                return "tool_node"
            else:
                print(
                    f"DB HELPER ROUTE destination=alt_tool_node "
                    f"loop={state['loop_count']} tool_calls={tool_names}",
                    flush=True,
                )
                return "alt_tool_node"
        print(
            f"DB HELPER ROUTE destination=format_db_output "
            f"loop={state['loop_count']}",
            flush=True,
        )
        return "format_db_output"
    elif state["loop_count"] == LOOP_CONFIG["s-db"] if state["account_type"] == "Student" else LOOP_CONFIG["a-db"]:
        if getattr(last_message, "tool_calls", None):
            messages.append("Loop limit reached. Please provide the final answer without using any more tools.")
            return "db"
        else:
            return "format_db_output"
    else:
        tool_dump = []
        for message in messages:
            if isinstance(message, ToolMessage):
                tool_dump.append(message)
            elif isinstance(message, AIMessage):
                messages.append(AIMessage(content="Tool record only", tool_calls=message.tool_calls))
        messages.append("Failsafe: database agent broke the rules. Returning tool ressults instead.")
        messages.extend(tool_dump)
        return "format_db_output"

graph_builder = StateGraph(DatabaseHelperState, output_schema=DatabaseHelperOutput)

graph_builder.add_node("db", db_node)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("alt_tool_node", alt_tool_node)
graph_builder.add_node("format_db_output", format_db_output)

graph_builder.add_edge(START, "db")
graph_builder.add_conditional_edges("db", tool_route, ["tool_node", "alt_tool_node", "format_db_output"])
graph_builder.add_edge("tool_node", "db")
graph_builder.add_edge("alt_tool_node", "db")
graph_builder.add_edge("format_db_output", END)

db_graph = graph_builder.compile()
