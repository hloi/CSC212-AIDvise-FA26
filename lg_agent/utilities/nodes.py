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

import sys, os, time
    
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
from lg_agent.database_utils import async_get_student_basic_info
from data_pipeline.database.database_dev_tools import aconnect
import json

load_dotenv()

CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)
    CONTEXT_CONFIG = CONFIG["context_config"]
    LOOP_CONFIG = CONFIG["loop_limits"]

def _latest_user_text(messages) -> str:
    latest_user_text = next(
        (
            message.content.lower()
            for message in reversed(messages)
            if isinstance(message, HumanMessage) and isinstance(message.content, str)
        ),
        "",
    )

    return latest_user_text

def _needs_student_major_lookup(messages) -> bool:
    latest_user_text = _latest_user_text(messages)
    return any(
        phrase in latest_user_text
        for phrase in (
            "my major",
            "my program",
            "program of study",
            "what major am i",
            "which major am i",
            "what program am i",
            "which program am i",
        )
    )

def _needs_student_course_history_lookup(messages) -> bool:
    latest_user_text = _latest_user_text(messages)
    return any(
        phrase in latest_user_text
        for phrase in (
            "course history",
            "my transcript",
            "courses have i taken",
            "course have i taken",
            "classes have i taken",
            "class have i taken",
            "courses i've taken",
            "courses i have taken",
            "classes i've taken",
            "classes i have taken",
            "courses did i take",
            "classes did i take",
            "courses have i completed",
            "classes have i completed",
        )
    )

def _normalize_flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)

def _normalize_request(value) -> str:
    if value is None or isinstance(value, bool):
        return ""
    request = str(value).strip()
    if request.lower() in {"false", "none", "null", "no", "n/a"}:
        return ""
    return request

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

    if (
        CONTEXT_CONFIG["s-planner"]["context-select"] == "full"
        and not _needs_student_course_history_lookup(state["messages"])
        and _needs_student_major_lookup(state["messages"])
    ):
        async with aconnect() as conn:
            student_info = await async_get_student_basic_info(conn, state["user_id"])
        programs = student_info.get("ProgramsOfStudy", [])
        program_titles = [
            program["Title"]
            for program in programs
            if isinstance(program, dict) and program.get("Title")
        ]
        if program_titles:
            if len(program_titles) == 1:
                answer = (
                    f"Your program of study is '{program_titles[0]}'. "
                    "Your profile does not list a separate declared major."
                )
            else:
                titles = ", ".join(f"'{title}'" for title in program_titles)
                answer = (
                    f"Your programs of study are {titles}. "
                    "Your profile does not list a separate declared major."
                )
        else:
            answer = "Your profile does not list a program of study or a declared major."
        state["plan"] = {
            "requires_database": False,
            "requires_web_search": False,
            "requires_insertion": False,
            "answer": answer,
            "info_needed_db": "",
            "info_needed_web": "",
            "info_to_insert": "",
        }
        print("STUDENT PLAN:", state["plan"], flush=True)
        return state

    structured_llm = planning_llm.with_structured_output(SPlanSchema)

    c_level = CONTEXT_CONFIG["s-planner"]["context-select"]
    system_prompt = CONTEXT_CONFIG["s-planner"]["context-level"][c_level]

    planner_rules = """
    Use the authenticated student's records for questions about their
    program, courses, grades, credits, advisor, or other academic information.

    Before requesting a database lookup, review the Database Results already
    provided in this planning run. Request only information that is still
    missing. If a result answers the question, provide a final answer and set
    all requires_* fields to false. Do not repeat the same database request.

    Treat questions about existing records as read-only. Request an insertion
    only when the student explicitly asks to save or change their own
    permitted profile information. Never infer an update from a question.
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

    planner_started_at = time.perf_counter()
    print("STUDENT PLANNER LLM CALL START", flush=True)
    try:
        planner_result = await structured_llm.ainvoke(
            messages,
            config=modified_config,
        )
    except Exception:
        elapsed = time.perf_counter() - planner_started_at
        print(
            f"STUDENT PLANNER LLM CALL FAILED after {elapsed:.2f}s",
            flush=True,
        )
        raise
    elapsed = time.perf_counter() - planner_started_at
    print(f"STUDENT PLANNER LLM CALL END after {elapsed:.2f}s", flush=True)
    response = planner_result.model_dump()

    # Correct inconsistent structured output from the model. 
    db_request = _normalize_request(response.get("info_needed_db"))
    web_request = _normalize_request(response.get("info_needed_web"))
    insertion_request = _normalize_request(response.get("info_to_insert"))

    response["requires_database"] = _normalize_flag(response.get("requires_database"))
    response["requires_web_search"] = _normalize_flag(response.get("requires_web_search"))
    if "requires_insertion" in response:
        response["requires_insertion"] = _normalize_flag(response.get("requires_insertion"))
    response["info_needed_db"] = db_request
    response["info_needed_web"] = web_request
    if "info_to_insert" in response:
        response["info_to_insert"] = insertion_request

    if db_request:
        response["requires_database"] = True
        response["answer"] = ""

    if web_request:
        response["requires_web_search"] = True
        response["answer"] = ""

    if insertion_request:
        response["requires_insertion"] = True

    if (
        c_level == "full"
        and not state.get("db_info")
        and _needs_student_major_lookup(state["messages"])
    ):
        response["requires_database"] = True
        response["requires_web_search"] = False
        response["requires_insertion"] = False
        response["answer"] = ""
        response["info_needed_web"] = None
        response["info_to_insert"] = ""
        response["info_needed_db"] = (
            "Look up the authenticated student's basic profile and identify "
            "their program(s) of study and declared major."
        )

    if (
        c_level == "full"
        and not state.get("db_info")
        and _needs_student_course_history_lookup(state["messages"])
    ):
        response["requires_database"] = True
        response["requires_web_search"] = False
        response["requires_insertion"] = False
        response["answer"] = ""
        response["info_needed_web"] = None
        response["info_to_insert"] = ""
        response["info_needed_db"] = (
            "Use the student_course_history tool to retrieve the authenticated "
            "student's completed and attempted courses, including course codes, "
            "titles, and grades. This is a read-only lookup; do not insert or "
            "change any student records."
        )

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

    db_started_at = time.perf_counter()
    print(
        f"STUDENT DB LLM CALL START loop={state['loop_count']}",
        flush=True,
    )
    try:
        result = await llm_with_db_tools.ainvoke(messages, config=modified_config)
    except Exception:
        elapsed = time.perf_counter() - db_started_at
        print(
            f"STUDENT DB LLM CALL FAILED after {elapsed:.2f}s "
            f"loop={state['loop_count']}",
            flush=True,
        )
        raise
    elapsed = time.perf_counter() - db_started_at
    tool_names = [
        tool_call.get("name", "unknown")
        for tool_call in getattr(result, "tool_calls", [])
    ]
    print(
        f"STUDENT DB LLM CALL END after {elapsed:.2f}s "
        f"loop={state['loop_count']} tool_calls={tool_names}",
        flush=True,
    )

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
