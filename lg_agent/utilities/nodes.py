"""
Copyright 2026 Luca Silver

Core planning and helper nodes for both student and advisor chat agents.

Planning Nodes:
- `s_planner_node`: Student planner that decides which sources (database, web, insertion) are needed and what info to gather from them (or what info to insert) for the current user input.
- `a_planner_node`: Advisor planner that decides what sources (database, web) are needed and what info to gather from them for the current user input.

Helper Nodes:
- `db_node`: Database helper that formulates and executes database queries using available tools.
- `web_node`: Web search helper that formulates and executes web searches using available tools.
- `insertion_node`: Insertion helper that processes student profile updates (interests, tracked sections).

These nodes are integrated into LangGraph agents to provide multi-turn planning and tool execution workflows.
System prompts and loop limits for each node are defined in config.json and can be adjusted to control agent behavior and prevent infinite loops.
"""

import sys, os
    
# adds utilities directory to system path if not already there
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# adds root directory to system path if not already there
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_customize_config
from utilities.state import APlannerState, SPlannerState, DatabaseHelperState, WebSearchHelperState, InsertionHelperState
from utilities.schemas import APlanSchema, SPlanSchema
from utilities.tools import db_tools, web_tools, insertion_tools, alt_db_tools
from utilities.model_inits import db_llm, planning_llm, web_llm, insertion_llm
from utilities.TestModel import FakeChatModel
import json

load_dotenv()

CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)
    CONTEXT_CONFIG = CONFIG["context_config"]
    LOOP_CONFIG = CONFIG["loop_limits"]

async def s_planner_node(state: SPlannerState, config: RunnableConfig) -> SPlannerState:
    """
    Builds the next plan for the student-facing agent.

    This node collects the current conversation plus any database, web, or insertion
    results that have already been gathered, then asks the planning model to decide
    whether more tool use is required. The returned plan is stored in state and used
    by the graph router to decide the next branch.

    Args:
        state (SPlannerState): Current student-agent state containing messages,
            loop counter, prior tool results, and user metadata.

    Returns:
        SPlannerState: Updated state with a new "plan" value from the planning model.

    Side Effects:
        - Increments the loop counter.
        - May initialize empty db_info and web_info lists.
    """

    state["loop_count"] += 1
    state["plan"] = None

    structured_llm = planning_llm.with_structured_output(SPlanSchema)

    c_level = CONTEXT_CONFIG["s-planner"]["context-select"]
    system_prompt = CONTEXT_CONFIG["s-planner"]["context-level"][c_level]

    planner_rules = """
    You have access to the authenticated student's academic records.

    For questions about the student's major, program, GPA, credits, advisor,
    graduation, interests, tracked sections, or course history, request database
    information before answering.

    For "What is my major?", set requires_database to true and set
    info_needed_db to: "Call student_basic_info for the current authenticated
    student and return ProgramsOfStudy." The current student ID is already
    available in the agent state. Do not request the student ID.

    Never insert a student ID, major, program, GPA, credits, advisor, or other
    existing academic record. Database insertion is only for new interests or
    goals explicitly stated by the student.

    Do not claim that academic records are unavailable when a student ID is
    present.

    Questions such as "What is my major?" and "What courses have I taken?" are
    read-only questions and must never request insertion.
    """

    messages = [
        SystemMessage(
            content=(
                system_prompt
                + "\n"
                + planner_rules
                + "\nLoop limit = "
                + str(LOOP_CONFIG["s-planner"])
            )
        )
    ]

    messages.extend(state["messages"])

    if "db_info" not in state:
        state["db_info"] = []
    else:
        for query_result in state["db_info"]:
            messages.append(
                HumanMessage(
                    content=(
                        f"Database Query: {query_result['query']}\n"
                        f"Database Result: {query_result['result']}"
                    )
                )
            )

    if "web_info" not in state:
        state["web_info"] = []
    else:
        for query_result in state["web_info"]:
            messages.append(
                HumanMessage(
                    content=(
                        f"Web Search Query: {query_result['query']}\n"
                        f"Web Search Result: {query_result['result']}"
                    )
                )
            )

    if state.get("insertion_result"):
        messages.append(
            HumanMessage(
                content=(
                    "Result of last insertion attempt: "
                    + state["insertion_result"]
                )
            )
        )

    if c_level != "no-tools":
        messages.append(
            HumanMessage(
                content="Current loop count = " + str(state["loop_count"])
            )
        )

    modified_config = copilotkit_customize_config(
        config,
        emit_messages=False,
        emit_tool_calls=False,
    )

    response = (
        await structured_llm.ainvoke(
            messages,
            config=modified_config,
        )
    ).model_dump()

    response = (
        await structured_llm.ainvoke(
            messages,
            config=modified_config,
        )
    ).model_dump()


    # Correct inconsistent structured output from the model. 
    db_request = (response.get("info_needed_db") or "").strip()
    web_request = (response.get("info_needed_web") or "").strip()
    insertion_request = (response.get("info_to_insert") or "").strip()

    if db_request:
        response["requires_database"] = True
        response["answer"] = ""

    if web_request:
        response["requires_web_search"] = True
        response["answer"] = ""

    if insertion_request:
        response["requires_insertion"] = True

    print("STUDENT PLAN:", response, flush=True)

    state["plan"] = response
    return state

async def a_planner_node(state: APlannerState, config: RunnableConfig) -> APlannerState:
    """
    Builds the next plan for the advisor-facing agent.

    This node is the advisor counterpart to the student planner. It gathers the
    current conversation, prior database and web results, and loop context, then
    asks the planning model to decide whether more tool use is needed before a final
    answer can be produced.

    Args:
        state (APlannerState): Current advisor-agent state containing messages,
            loop counter, prior tool results, and user metadata.

    Returns:
        APlannerState: Updated state with a new "plan" value from the planning model.

    Side Effects:
        - Increments the loop counter.
        - May initialize empty db_info and web_info lists.
    """

    state["loop_count"] += 1
    
    structured_llm = planning_llm.with_structured_output(APlanSchema)

    c_level = CONTEXT_CONFIG["a-planner"]["context-select"]
    system_prompt = CONTEXT_CONFIG["a-planner"]["context-level"][c_level]

    messages = []
    messages.extend(state["messages"])

    if len(messages) == 1:
        messages.append(SystemMessage(content=system_prompt + ("Loop limit = " + str(LOOP_CONFIG["a-planner"]) if CONTEXT_CONFIG["a-planner"] != "no-tools" else "")))

    if "db_info" not in state:
        state["db_info"] = []
    else:
        for QueryResult in state["db_info"]:
            messages.append(HumanMessage(content=f"Database Query: {QueryResult['query']}\nDatabase Result: {QueryResult['result']}"))
    if "web_info" not in state:
        state["web_info"] = []
    else:
        for QueryResult in state["web_info"]:
            messages.append(HumanMessage(content=f"Web Search Query: {QueryResult['query']}\nWeb Search Result: {QueryResult['result']}"))

    if CONTEXT_CONFIG["a-planner"] != "no-tools":
        messages.append(HumanMessage(content="Current loop count = " + str(state["loop_count"])))

    modified_config = copilotkit_customize_config(
        config,
        emit_messages=False,
        emit_tool_calls=False 
    )

    response = (await structured_llm.ainvoke(messages, config=modified_config)).model_dump()

    # Temporary safety guard: disable database insertion.
    # Read-only questions should never enter the insertion graph.
    response["requires_insertion"] = False
    response["info_to_insert"] = ""

    print("STUDENT PLAN:", response, flush=True)

    state["plan"] = response
    return state

async def db_node(state: DatabaseHelperState, config: RunnableConfig):
    """
    Executes the database helper model with the appropriate tool set.

    The node chooses between student and advisor database tools based on the user's 
    account type, seeds the message list with system guidance when this is the first turn, 
    preserves prior tool messages, and sends the curated conversation to the database LLM. 
    The LLM may decide to call one or more database tools before producing its response.

    Args:
        state (DatabaseHelperState): Helper state containing the requested information,
            message history, loop counter, user ID, and account type.

    Returns:
        dict: A partial state update containing a single AI response message under
            the "messages" key.

    Side Effects:
        - Increments the loop counter.
        - May append an initial system prompt and task description to state messages.
    """

    state["loop_count"] += 1

    if state["account_type"] == "Student":
        llm_with_db_tools = (
            db_llm.bind_tools(db_tools, runtime_state=state)
            if isinstance(db_llm, FakeChatModel)
            else db_llm.bind_tools(db_tools)
        )
        c_level = CONTEXT_CONFIG["s-db"]["context-select"]
        system_prompt = CONTEXT_CONFIG["s-db"]["context-level"][c_level]
    else:
        llm_with_db_tools = (
            db_llm.bind_tools(alt_db_tools, runtime_state=state)
            if isinstance(db_llm, FakeChatModel)
            else db_llm.bind_tools(alt_db_tools)
        )
        c_level = CONTEXT_CONFIG["a-db"]["context-select"]
        system_prompt = CONTEXT_CONFIG["a-db"]["context-level"][c_level]
    
    # if state messages is empty add a message with the info needed, otherwise pass the messages through
    if len(state["messages"]) == 0:
        state["messages"].append(SystemMessage(content=system_prompt + "Loop limit = " + str(LOOP_CONFIG["s-db"] if state["account_type"] == "Student" else LOOP_CONFIG["a-db"])))
        state["messages"].append(HumanMessage(content=f"The planning node has determined that the following information is needed from the database to answer the user's question: {state['info_needed']}"))
    
    messages = []
    messages.extend(state["messages"][:2])
    if len(state["messages"]) > 2:
        for message in state["messages"][2:-2]:
            if isinstance(message, ToolMessage):
                messages.append(message)
            elif isinstance(message, AIMessage):
                messages.append(AIMessage(content="Tool record only", tool_calls=message.tool_calls))
        messages.extend(state["messages"][-2:])
    
    messages.append(HumanMessage(content="Current loop count = " + str(state["loop_count"])))

    modified_config = copilotkit_customize_config(
        config,
        emit_messages=False,
        emit_tool_calls=False 
    )

    result = await llm_with_db_tools.ainvoke(messages, config=modified_config)

    return {"messages": [result]}

async def web_node(state: WebSearchHelperState, config: RunnableConfig):
    """
    Executes the web search helper model with the search tool set.

    The node prepares the message history for the web LLM, including an initial
    system prompt and search goal on the first pass. It preserves prior tool records
    and sends the curated conversation to the model so it can search the web or
    summarize the gathered results.

    Args:
        state (WebSearchHelperState): Helper state containing the requested web
            information, message history, and loop counter.

    Returns:
        dict: A partial state update containing a single AI response message under
            the "messages" key.

    Side Effects:
        - Increments the loop counter.
        - May append an initial system prompt and task description to state messages.
    """
    
    state["loop_count"] += 1

    llm_with_web_tools = (
        web_llm.bind_tools(web_tools, runtime_state=state)
        if isinstance(web_llm, FakeChatModel)
        else web_llm.bind_tools(web_tools)
    )

    c_level = CONTEXT_CONFIG["web"]["context-select"]
    system_prompt = CONTEXT_CONFIG["web"]["context-level"][c_level]

    # if state messages is empty add a message with the info needed, otherwise pass the messages through
    if len(state["messages"]) == 0:
        state["messages"].append(SystemMessage(content=system_prompt + "Loop limit = " + str(LOOP_CONFIG["web"])))
        state["messages"].append(HumanMessage(content=f"The planning node has determined that the following information is needed from the web to answer the user's question: {state['info_needed']}"))

    messages = []
    messages.extend(state["messages"][:2])
    if len(state["messages"]) > 2:
        for message in state["messages"][2:-2]:
            if isinstance(message, ToolMessage):
                messages.append(message)
            elif isinstance(message, AIMessage):
                messages.append(AIMessage(content="Tool record only", tool_calls=message.tool_calls))
        messages.extend(state["messages"][-2:])
    
    messages.append(HumanMessage(content="Current loop count = " + str(state["loop_count"])))

    modified_config = copilotkit_customize_config(
        config,
        emit_messages=False,
        emit_tool_calls=False 
    )

    result = await llm_with_web_tools.ainvoke(messages, config=modified_config)

    return {"messages": [result]}

async def insertion_node(state: InsertionHelperState, config: RunnableConfig) -> InsertionHelperState:
    """
    Executes the insertion helper model to persist new student information.

    This node prepares a tool-enabled conversation for the insertion LLM, which is
    used to decide whether any new student information should be written into the
    database. On the first pass it seeds the conversation with instructions that
    describe what should be inserted.

    Args:
        state (InsertionHelperState): Helper state containing the information to
            insert, message history, loop counter, and user ID.

    Returns:
        dict: A partial state update containing a single AI response message under
            the "messages" key.

    Side Effects:
        - Increments the loop counter.
        - May append an initial system prompt and insertion request to state messages.
    """

    state["loop_count"] += 1

    llm_with_insertion_tools = (
        insertion_llm.bind_tools(insertion_tools, runtime_state=state)
        if isinstance(insertion_llm, FakeChatModel)
        else insertion_llm.bind_tools(insertion_tools)
    )

    c_level = CONTEXT_CONFIG["insertion"]["context-select"]
    system_prompt = CONTEXT_CONFIG["insertion"]["context-level"][c_level]

    # if state messages is empty add a message with the info to be inserted, otherwise pass the messages through
    if len(state["messages"]) == 0:
        state["messages"].append(SystemMessage(content=system_prompt + "Loop limit = " + str(LOOP_CONFIG["insertion"])))
        state["messages"].append(HumanMessage(content=f"The planning node has determined that the following information about the student should be added into the database if it is not already present: {state['info_to_insert']}"))
    
    messages = []
    messages.extend(state["messages"][:2])
    if len(state["messages"]) > 2:
        for message in state["messages"][2:-2]:
            if isinstance(message, ToolMessage):
                messages.append(message)
            elif isinstance(message, AIMessage):
                messages.append(AIMessage(content="Tool record only", tool_calls=message.tool_calls))
        messages.extend(state["messages"][-2:])
    
    messages.append(HumanMessage(content="Current loop count = " + str(state["loop_count"])))

    modified_config = copilotkit_customize_config(
        config,
        emit_messages=False,
        emit_tool_calls=False 
    )

    result = await llm_with_insertion_tools.ainvoke(messages, config=modified_config)

    return {"messages": [result]}
