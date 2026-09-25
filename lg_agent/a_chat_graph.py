"""
Copyright 2026 Luca Silver

Advisor-facing chat agent that routes advisor queries through planning and helper graphs.

Functions:
- `invoke_db_helper`: Invokes the database helper graph and merges results back to planner state.
- `invoke_web_helper`: Invokes the web search helper graph and merges results back to planner state.
- `route_from_planning`: Routes the planner's decision to appropriate helper graphs or answer node. Enforces a maximum number of loops before forcing an answer.
- `answer_node`: Appends the final planner response to the conversation message history.

Graph Structure:
- START -> planning
- planning -> [invoke_db_helper | invoke_web_helper | answer_node] (conditional) (cam be parallel)
- invoke_db_helper -> planning (feedback loop) (deferred)
- invoke_web_helper -> planning (feedback loop) (deferred)
- answer_node -> END

Exports:
- `a_chat_graph`: Compiled LangGraph advisor chat agent.

Loop limits are defined in config.json.
"""

import sys, os

# adds lg_agent directory to system path if not already there
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from lg_agent.utilities.state import APlannerState
from langchain_core.messages import AIMessage
from lg_agent.utilities.nodes import a_planner_node
from db_helper_graph import db_graph
from web_helper_graph import web_graph
import json

# cd my-agent && .venv\Scripts\activate && npx @langchain/langgraph-cli dev --port 8123 --no-browser
load_dotenv()

CONFIG_PATH = os.path.join(PARENT_DIR, "config.json")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)
    LOOP_CONFIG = CONFIG["loop_limits"]

def _extract_plan(state: APlannerState) -> tuple[dict, dict]:
    """
    Safely extracts planner output and nested content dict from state.

    Some model/tooling paths may produce plan payloads where "content" is a
    string or missing entirely. This helper normalizes both objects so routing
    code can perform key lookups without type errors.
    """
    plan = state.get("plan", {})
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except Exception:
            return {}, {}
    if not isinstance(plan, dict):
        return {}, {}

    content = plan.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except Exception:
            content = {}
    if not isinstance(content, dict):
        content = {}

    return plan, content

def _plan_value(state: APlannerState, key: str, default=None):
    """
    Reads a planner field from either top-level plan or nested content.
    """
    plan, content = _extract_plan(state)
    if key in plan:
        return plan.get(key, default)
    return content.get(key, default)

async def invoke_db_helper(state: APlannerState):
    """
    Invokes the advisor database helper graph and merges its result back into state.

    The advisor planner may request database information about a student or course.
    This node forwards that request to the database helper graph, then appends the
    returned query/result pair to the accumulated db_info list.

    Args:
        state (APlannerState): Current planner state containing the selected database
            request in plan["info_needed_db"].

    Returns:
        dict: Partial state update with the updated "db_info" list.
    """
    info_needed = _plan_value(state, "info_needed_db")
    if not info_needed:
        return {"db_info": state.get("db_info", [])}
    
    db_helper_state = {"info_needed": info_needed, "messages": [], "loop_count": 0, "user_id": state["user_id"], "account_type": "Advisor"}
    result = await db_graph.ainvoke(db_helper_state)
    db_info = state.get("db_info", [])
    db_info.append(result.get("info", {}))
    return {"db_info": db_info}

async def invoke_web_helper(state: APlannerState):
    """
    Invokes the web helper graph and merges its result back into state.

    The advisor planner may request web information to support an answer. This node
    forwards that request to the web helper graph, then appends the returned
    query/result pair to the accumulated web_info list.

    Args:
        state (APlannerState): Current planner state containing the selected web
            request in plan["info_needed_web"].

    Returns:
        dict: Partial state update with the updated "web_info" list.
    """
    info_needed = _plan_value(state, "info_needed_web")
    if not info_needed:
        return {"web_info": state.get("web_info", [])}
    
    web_helper_state = {"info_needed": info_needed, "messages": [], "loop_count": 0}
    result = await web_graph.ainvoke(web_helper_state)
    web_info = state.get("web_info", [])
    web_info.append(result.get("info", {}))
    return {"web_info": web_info}

def route_from_planning(state: APlannerState):
    """
    Chooses which helper graphs to run after planning.

    The planner can request database lookups or web lookups, or provide a final
    answer directly. This router converts the plan into one or more graph sends.
    When both database and web information are requested, they are dispatched in
    parallel. If no tools are needed, control routes directly to the answer node.

    Args:
        state (APlannerState): Current planner state containing the loop counter and
            the structured plan produced by the planning node.

    Returns:
        list[Send]: One or more graph sends describing the next execution branch.
    """
    requires_database = bool(_plan_value(state, "requires_database", False))
    requires_web_search = bool(_plan_value(state, "requires_web_search", False))
    answer = _plan_value(state, "answer", "")

    if state["loop_count"] < LOOP_CONFIG["a-planner"]:
        routes = []
        if requires_database:
            routes.append("invoke_db_helper")
        if requires_web_search:
            routes.append("invoke_web_helper")
        if not routes:
            if answer is not None and answer != "":
                routes.append("answer_node")
            else:
                messages = state["messages"] + [
                    AIMessage(content="No information was selected and no answer was provided. Re-evaluate and provide either tool requirements or a final answer.")
                ]
                return [Send("planning", {**state, "messages": messages})]
        return [Send(route, state) for route in routes]
    elif answer is not None and answer != "":
        return [Send("answer_node", state)]
    else:
        if state["loop_count"] == LOOP_CONFIG["a-planner"]:
            messages = state["messages"] + [AIMessage(content="You went over the loop limit. Give your final answer now.")]
            return [Send("planning", {"messages": messages, "loop_count": state["loop_count"] + 1})]
        else:
            messages = AIMessage(content="Failsafe: AI broke the rules. Please try to rephrase your question.")
            return [Send("answer_node", {"messages": [messages]})]

def answer_node(state: APlannerState) -> APlannerState:
    """
    Appends the final planner answer to the conversation state.

    Args:
        state (APlannerState): Current planner state containing the final answer in
            plan["answer"].

    Returns:
        APlannerState: Updated state with the final AI answer appended to messages.
    """
    answer = _plan_value(state, "answer")
    if answer is None or answer == "":
        answer = "I couldn't produce a final answer for that request. Please rephrase your question and try again."
    # return {"messages": state["messages"] + [AIMessage(content=answer)]}
    return {"messages": [AIMessage(content=answer)]}

graph_builder = StateGraph(APlannerState)

graph_builder.add_node("planning", a_planner_node, defer=True)
graph_builder.add_node("invoke_db_helper", invoke_db_helper)
graph_builder.add_node("invoke_web_helper", invoke_web_helper)
graph_builder.add_node("answer_node", answer_node)

graph_builder.add_edge(START, "planning")
graph_builder.add_conditional_edges("planning", route_from_planning, ["invoke_db_helper", "invoke_web_helper", "answer_node"])
graph_builder.add_edge("invoke_db_helper", "planning")
graph_builder.add_edge("invoke_web_helper", "planning")
graph_builder.add_edge("answer_node", END)

a_chat_graph = graph_builder.compile()
