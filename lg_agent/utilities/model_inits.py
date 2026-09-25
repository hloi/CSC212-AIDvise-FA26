"""
Copyright 2026 Luca Silver

Initializes language models for different agent nodes and modes (production or testing).

Functions:
- `_load_model_config`: Loads model configuration from model_select.json.
- `_normalize_model_name`: Normalizes model names by converting to lowercase and replacing hyphens with underscores.
- `_s_planner_tool_testing_model`: Creates a fake testing model for the student planner node that calls all helper agents.
- `_a_planner_tool_testing_model`: Creates a fake testing model for the advisor planner node that calls all helper agents.
- `_planner_loop_testing_model`: Creates a fake testing model for the planner nodes to test loop limits.
- `_s_db_tool_testing_model`: Creates a fake testing model for the student database helper node that test all its tools.
- `_a_db_tool_testing_model`: Creates a fake testing model for the advisor database helper node that test all its tools.
- `_db_loop_testing_model`: Creates a fake testing model for the database helper nodes to test loop limits.
- `_db_do_nothing_testing_model`: Creates a fake testing model for the database helper nodes that does nothing to help test planner loop limits.
- `_web_tool_testing_model`: Creates a fake testing model for the web helper node that tests all its tools.
- `_web_loop_testing_model`: Creates a fake testing model for the web helper nodes to test loop limits.
- `_web_do_nothing_testing_model`: Creates a fake testing model for the web helper nodes that does nothing.
- `_insertion_tool_testing_model`: Creates a fake testing model for the insertion helper node that tests all its tools.
- `_insertion_loop_testing_model`: Creates a fake testing model for the insertion helper node to test loop limits.
- `_insertion_do_nothing_testing_model`: Creates a fake testing model for the insertion helper node that does nothing.
- `_alerts_testing_model`: Creates a fake testing model for the alerts agent node.
- `_create_model`: Factory function that creates appropriate chat models based on model name and node type.

Modules Initialized:
- `planning_llm`: Language model for planning nodes.
- `db_llm`: Language model for database helper nodes.
- `web_llm`: Language model for web search helper nodes.
- `insertion_llm`: Language model for insertion helper nodes.
- `alerts_llm`: Language model for alerts agent nodes.
"""

import os, sys

# adds utilities directory to system path if not already there
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# adds root directory to system path if not already there
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolCall
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from utilities.TestModel import FakeChatModel

import json

load_dotenv()

def _load_model_config() -> tuple[dict, str]:
    """
    Load and return model selection configuration.

    Reads `config.json` from the repository root and extracts the
    `model_select` section. Returns a tuple `(models, mode)` where
    `models` is the dictionary of model choices for the active mode
    and `mode` is the active mode string (e.g. "production", "testing").

    Returns:
        tuple[dict, str]: (models, mode)
    Raises:
        FileNotFoundError: if `config.json` does not exist.
        KeyError: if expected keys are missing from the config.
    """
    with open(os.path.join(ROOT_DIR, "config.json"), "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
        MODEL_SELECT = CONFIG.get("model_select")
    mode = MODEL_SELECT.get("mode")
    models = MODEL_SELECT.get(mode)
    return models, mode

def _normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name string for internal matching.

    Performs a simple normalization by trimming whitespace, converting
    to lowercase, and replacing hyphens with underscores. This ensures
    configuration names like "sonnet-4-6" match internal case labels
    such as "sonnet_4_6".

    Args:
        model_name (str): The model name from configuration.

    Returns:
        str: The normalized model name.
    """
    return model_name.strip().lower().replace("-", "_")

def _s_planner_tool_testing_model() -> FakeChatModel:
    """
    Create a fake chat model for student planning tool tests.

    The returned `FakeChatModel` yields a sequence of
    `AIMessage`s that instruct the planner to exercise database,
    web, and insertion helper tools, followed by completion/fallback
    messages. This is used in unit/integration tests to simulate a
    planning model that requests helper tool calls.

    Returns:
        FakeChatModel: a test model instance that produces messages
            and tool call instructions.

    Note: This should be used with student accounts only.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=json.dumps(
                        {
                            "requires_database": True,
                            "requires_web_search": True,
                            "requires_insertion": True,
                            "info_needed_db": (
                                "Run a thorough test of all tools."
                            ),
                            "info_needed_web": (
                                "Run a thorough test of all tools."
                            ),
                            "info_to_insert": (
                                "Run a thorough test of all tools."
                            )
                        }
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {
                            "answer": (
                                "Test completed."
                            )
                        }
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {
                            "answer": (
                                "Testing fallback: no additional tool calls are needed."
                            )
                        }
                    )
                ),
            ]
        )
    )

def _a_planner_tool_testing_model() -> FakeChatModel:
    """
    Create a fake chat model for advisor planning tool tests.

    Similar to `_s_planner_tool_testing_model` but tailored for the
    advisor/planner flow. Produces messages that request database and
    web helper tool usage, followed by completion and fallback
    responses for testing.

    Returns:
        FakeChatModel: a test model instance for advisor planning.
    
    Note: This should be used with advisor accounts only.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=json.dumps(
                        {
                            "requires_database": True,
                            "requires_web_search": True,
                            "info_needed_db": (
                                "Run a thorough test of all tools."
                            ),
                            "info_needed_web": (
                                "Run a thorough test of all tools."
                            )
                        }
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {
                            "answer": (
                                "Test completed."
                            ),
                        }
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {
                            "answer": (
                                "Testing fallback: no additional tool calls are needed."
                            ),
                        }
                    )
                ),
            ]
        )
    )

def _planner_loop_testing_model() -> FakeChatModel:
    """
    Create a fake planner model that repeatedly requests database help.

    This test model produces multiple messages that continue to request
    database information across several loops. It is intended to
    validate planner loop limits and ensure the controller stops after
    a configured number of iterations.

    Returns:
        FakeChatModel: a loop-testing model instance.

    Note: This should be used with the planner loop limit set to 3 for whichever version of the planner is being tested.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=json.dumps(
                        {
                            "requires_database": True,
                            "info_needed_db": (
                                "Do nothing. Current loop 1."
                            )
                        }
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {
                            "requires_database": True,
                            "info_needed_db": (
                                "Do nothing. Current loop 2."
                            )
                        }
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {
                            "requires_database": True,
                            "info_needed_db": (
                                "Do nothing. Current loop 3."
                            )
                        }
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {
                            "requires_database": True,
                            "info_needed_db": (
                                "Do nothing. Current loop 4."
                            )
                        }
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {
                            "answer": (
                                "Should have stopped by now. Loop limit may be broken. Ensure it is set to 3 for this test."
                            )
                        }
                    )
                ),
            ]
        )
    )

def _s_db_tool_testing_model() -> FakeChatModel:
    """
    Create a fake student-database helper model that exercises DB tools.

    The returned `FakeChatModel` contains an initial message
    that includes a long list of `ToolCall` entries exercising the
    database helper API (queries, filters, edge cases). Follow-up
    messages indicate completion and fallback behavior. Use this model
    in tests to validate the DB helper tooling and error handling.

    Returns:
        FakeChatModel: a test model instance for student DB tools.

    Note: This should be used with ``TestDB.db`` configured as the database and with John Doe as the user.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Comprehensive database-tool smoke test: exercise every student DB tool and edge cases.",
                    tool_calls=[
                        # check current time
                        ToolCall(name="get_current_time", args={}, id="ct"),

                        # get department list
                        ToolCall(name="get_department_list", args={}, id="dl"),

                        # get departments in category
                        ToolCall(name="get_departments_in_category", args={"category": "Behavioral Science Elective"}, id="dc1"),
                        ToolCall(name="get_departments_in_category", args={"category": "Humanities Elective"}, id="dc2"),
                        ToolCall(name="get_departments_in_category", args={"category": "Mathematics Elective"}, id="dc3"),
                        ToolCall(name="get_departments_in_category", args={"category": "Science Elective"}, id="dc4"),
                        ToolCall(name="get_departments_in_category", args={"category": "Lab Science Elective"}, id="dc5"),
                        ToolCall(name="get_departments_in_category", args={"category": "Social Sciences Elective"}, id="dc6"),
                        ToolCall(name="get_departments_in_category", args={"category": "Liberal Arts Elective"}, id="dc7"),
                        ToolCall(name="get_departments_in_category", args={"category": "General Elective"}, id="dc8"),
                        ToolCall(name="get_departments_in_category", args={"category": "General Education"}, id="dc9"),
                        ToolCall(name="get_departments_in_category", args={"category": "GenEd"}, id="dc10"),
                        ToolCall(name="get_departments_in_category", args={"category": "Nonexistent Category"}, id="dc11"),

                        # query for real course by code and title
                        ToolCall(name="course_query_by_code", args={"course_code": "CSC 212"}, id="cqc1"),
                        ToolCall(name="course_query_by_code", args={"course_code": "CSC212"}, id="cqc2"),
                        ToolCall(name="course_query_by_title", args={"course_title": "Intro to Software Engineering"}, id="cqt1"),

                        # query for co-op
                        ToolCall(name="course_query_by_title", args={"course_title": "Cooperative Work Experience"}, id="cqt2"),

                        # query for nonexistent course by code and title
                        ToolCall(name="course_query_by_code", args={"course_code": "XXX 999"}, id="cqc3"),
                        ToolCall(name="course_query_by_code", args={"course_code": "CSCI 212"}, id="cqc4"),
                        ToolCall(name="course_query_by_code", args={"course_code": "CSCI212"}, id="cqc5"),
                        ToolCall(name="course_query_by_title", args={"course_title": "Nonexistent Course Title"}, id="cqt3"),

                        # basic test course filter
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "departments": ["CSC"],
                                }
                            },
                            id="cf1",
                        ),

                        # test course filter no terms
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "departments": ["CSC"],
                                }
                            },
                            id="cf2",
                        ),

                        # test course filter with course level
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "departments": ["CSC"],
                                    "course_levels": [{"condition": ">", "level": "100"}, {"condition": "<", "level": "200"}],
                                }
                            },
                            id="cf3",
                        ),

                        # test course filter with credits
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "departments": ["CSC"],
                                    "credits": [{"condition": ">=", "credits": 3}, {"condition": "<=", "credits": 4}],
                                }
                            },
                            id="cf4",
                        ),

                        # test course filter only credits
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "credits": [{"condition": ">=", "credits": 3}, {"condition": "<=", "credits": 4}],
                                }
                            },
                            id="cf5",
                        ),

                        # test course filter with description keyword filter
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "keywords": ["software", "engineering"]
                                }
                            },
                            id="cf6",
                        ),

                        # test course filter with prerequisite keyword filter
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "prerequisites": ["CSC", "MAT"]
                                }
                            },
                            id="cf7",
                        ),

                        # test course filter with invalid filter keys (should be handled gracefully)
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "invalid_key": "invalid_value"
                                }
                            },
                            id="cf8",
                        ),

                        # test course filter with nonsense credits filter
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "credits": [{"condition": ">", "credits": 5}, {"condition": "<", "credits": 3}]
                                }
                            },
                            id="cf9",
                        ),

                        # test get course description for real course
                        ToolCall(name="get_course_description", args={"course_id": "1"}, id="cd1"),

                        # test get course description for nonexistent course
                        ToolCall(name="get_course_description", args={"course_id": "99999"}, id="cd2"),

                        # basic test section filter
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "course_codes": ["ACC 101", "ASL111"],
                                    "instructors": ["De Silva, Damindi", "Desmond, Eric", "Dunn, John D.,, Jr.", "Glover, Sabrina"],
                                }
                            },
                            id="sf1",
                        ),

                        # test section filter no terms
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "course_codes": ["ACC 101", "ASL111"],
                                    "instructors": ["De Silva, Damindi", "Desmond, Eric", "Dunn, John D.,, Jr.", "Glover, Sabrina"],
                                }
                            },
                            id="sf2",
                        ),

                        # test section filter no instructors
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "course_codes": ["ACC 101", "ASL111"],
                                }
                            },
                            id="sf3",
                        ),

                        # test section filter no course codes
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "instructors": ["De Silva, Damindi", "Desmond, Eric"],
                                }
                            },
                            id="sf4",
                        ),

                        # test section filter with teaching methods
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "course_codes": ["ACC 101", "ASL111"],
                                    "teaching_methods": ["Lecture", "Independent Study"],
                                }
                            },
                            id="sf5",
                        ),

                        # test section filter with credits
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "credits": [{"condition": "=", "credits": 5}, {"condition": ">", "credits": 3}],
                                }
                            },
                            id="sf6",
                        ),

                        # test section filters with enrollment capacity
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "enrollment_capacity": [{"condition": ">", "enrollment": 25}, {"condition": "<", "enrollment": 30}],
                                }
                            },
                            id="sf7",
                        ),

                        # test section filters with current enrollment
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "enrollment": [{"condition": ">", "enrollment": 0}, {"condition": "<", "enrollment": 3}],
                                }
                            },
                            id="sf8",
                        ),

                        # test section filters with location
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "location": ["MAIN Campus, Surprenant Hall, Classroom, 134", "Online"],
                                }
                            },
                            id="sf9",
                        ),

                        # test section filters with meet times
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "meet_times": [
                                        {"day": "Monday", "start_time": "08:00", "end_time": "9:15"},
                                        {"day": "Tuesday", "start_time": "02:00", "end_time": "15:15"}
                                    ],
                                }
                            },
                            id="sf10",
                        ),

                        # test section filter with invalid filter keys (should be handled gracefully)
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "invalid_key": "invalid_value"
                                }
                            },
                            id="sf11",
                        ),

                        # test get student basic info
                        ToolCall(name="student_basic_info", args={}, id="sbi"),

                        # test get student course history
                        ToolCall(name="student_course_history", args={}, id="sch"),

                        # test get student interests
                        ToolCall(name="student_interests", args={}, id="si"),

                        # test get student tracked sections
                        ToolCall(name="student_tracked_sections", args={}, id="sts"),

                        # test get program requirements with real program
                        ToolCall(name="program_requirements", args={"program_name": "Computer Science Transfer"}, id="pr1"),

                        # test get program requirements with nonexistent program
                        ToolCall(name="program_requirements", args={"program_name": "Imaginary Program"}, id="pr2"),

                        # test get upcoming events
                        ToolCall(name="upcoming_events", args={}, id="ue"),

                        # test get event dates for real event
                        ToolCall(name="event_dates", args={"event_name": "AI Career Panel"}, id="ed1"),

                        # test get event dates for nonexistent event
                        ToolCall(name="event_dates", args={"event_name": "Nonexistent Event"}, id="ed2"),
                    ],
                ),
                AIMessage(content="Database tool smoke test complete: all tools were exercised and edge cases tested."),
                AIMessage(content="Database fallback: no additional database actions are needed."),
            ]
        )
    )

def _a_db_tool_testing_model() -> FakeChatModel:
    """
    Create a fake advisor-database helper model that exercises DB tools.

    Similar to `_s_db_tool_testing_model` but with advisor-oriented IDs
    and scenarios (e.g., retrieving advisor students). Produces tool
    calls that exercise edge cases and error paths to validate the
    advisor DB helper implementation.

    Returns:
        FakeChatModel: a test model instance for advisor DB tools.

    Note: This should be used with ``TestDB.db`` configured as the database and with Bill Bebop as the user.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Comprehensive database-tool smoke test: exercise every advisor DB tool and edge cases.",
                    tool_calls=[
                        # check current time
                        ToolCall(name="get_current_time", args={}, id="ct"),

                        # get department list
                        ToolCall(name="get_department_list", args={}, id="dl"),

                        # get departments in category
                        ToolCall(name="get_departments_in_category", args={"category": "Behavioral Science Elective"}, id="dc1"),
                        ToolCall(name="get_departments_in_category", args={"category": "Humanities Elective"}, id="dc2"),
                        ToolCall(name="get_departments_in_category", args={"category": "Mathematics Elective"}, id="dc3"),
                        ToolCall(name="get_departments_in_category", args={"category": "Science Elective"}, id="dc4"),
                        ToolCall(name="get_departments_in_category", args={"category": "Lab Science Elective"}, id="dc5"),
                        ToolCall(name="get_departments_in_category", args={"category": "Social Sciences Elective"}, id="dc6"),
                        ToolCall(name="get_departments_in_category", args={"category": "Liberal Arts Elective"}, id="dc7"),
                        ToolCall(name="get_departments_in_category", args={"category": "General Elective"}, id="dc8"),
                        ToolCall(name="get_departments_in_category", args={"category": "General Education"}, id="dc9"),
                        ToolCall(name="get_departments_in_category", args={"category": "GenEd"}, id="dc10"),
                        ToolCall(name="get_departments_in_category", args={"category": "Nonexistent Category"}, id="dc11"),

                        # query for real course by code and title
                        ToolCall(name="course_query_by_code", args={"course_code": "CSC 212"}, id="cqc1"),
                        ToolCall(name="course_query_by_code", args={"course_code": "CSC212"}, id="cqc2"),
                        ToolCall(name="course_query_by_title", args={"course_title": "Intro to Software Engineering"}, id="cqt1"),

                        # query for co-op
                        ToolCall(name="course_query_by_title", args={"course_title": "Cooperative Work Experience"}, id="cqt2"),

                        # query for nonexistent course by code and title
                        ToolCall(name="course_query_by_code", args={"course_code": "XXX 999"}, id="cqc3"),
                        ToolCall(name="course_query_by_code", args={"course_code": "CSCI 212"}, id="cqc4"),
                        ToolCall(name="course_query_by_code", args={"course_code": "CSCI212"}, id="cqc5"),
                        ToolCall(name="course_query_by_title", args={"course_title": "Nonexistent Course Title"}, id="cqt3"),

                        # basic test course filter
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "departments": ["CSC"],
                                }
                            },
                            id="cf1",
                        ),

                        # test course filter no terms
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "departments": ["CSC"],
                                }
                            },
                            id="cf2",
                        ),

                        # test course filter with course level
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "departments": ["CSC"],
                                    "course_levels": [{"condition": ">", "level": "100"}, {"condition": "<", "level": "200"}],
                                }
                            },
                            id="cf3",
                        ),

                        # test course filter with credits
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "departments": ["CSC"],
                                    "credits": [{"condition": ">=", "credits": 3}, {"condition": "<=", "credits": 4}],
                                }
                            },
                            id="cf4",
                        ),

                        # test course filter only credits
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "credits": [{"condition": ">=", "credits": 3}, {"condition": "<=", "credits": 4}],
                                }
                            },
                            id="cf5",
                        ),

                        # test course filter with description keyword filter
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "keywords": ["software", "engineering"]
                                }
                            },
                            id="cf6",
                        ),

                        # test course filter with prerequisite keyword filter
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "prerequisites": ["CSC", "MAT"]
                                }
                            },
                            id="cf7",
                        ),

                        # test course filter with invalid filter keys (should be handled gracefully)
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "invalid_key": "invalid_value"
                                }
                            },
                            id="cf8",
                        ),

                        # test course filter with nonsense credits filter
                        ToolCall(
                            name="course_filter",
                            args={
                                "filters": {
                                    "credits": [{"condition": ">", "credits": 5}, {"condition": "<", "credits": 3}]
                                }
                            },
                            id="cf9",
                        ),

                        # test get course description for real course
                        ToolCall(name="get_course_description", args={"course_id": "1"}, id="cd1"),

                        # test get course description for nonexistent course
                        ToolCall(name="get_course_description", args={"course_id": "99999"}, id="cd2"),

                        # basic test section filter
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "course_codes": ["ACC 101", "ASL111"],
                                    "instructors": ["De Silva, Damindi", "Desmond, Eric", "Dunn, John D.,, Jr.", "Glover, Sabrina"],
                                }
                            },
                            id="sf1",
                        ),

                        # test section filter no terms
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "course_codes": ["ACC 101", "ASL111"],
                                    "instructors": ["De Silva, Damindi", "Desmond, Eric", "Dunn, John D.,, Jr.", "Glover, Sabrina"],
                                }
                            },
                            id="sf2",
                        ),

                        # test section filter no instructors
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "course_codes": ["ACC 101", "ASL111"],
                                }
                            },
                            id="sf3",
                        ),

                        # test section filter no course codes
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "instructors": ["De Silva, Damindi", "Desmond, Eric"],
                                }
                            },
                            id="sf4",
                        ),

                        # test section filter with teaching methods
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "terms": [{"year": 2026, "season": "Spring"}],
                                    "course_codes": ["ACC 101", "ASL111"],
                                    "teaching_methods": ["Lecture", "Independent Study"],
                                }
                            },
                            id="sf5",
                        ),

                        # test section filter with credits
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "credits": [{"condition": "=", "credits": 5}, {"condition": ">", "credits": 3}],
                                }
                            },
                            id="sf6",
                        ),

                        # test section filters with enrollment capacity
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "enrollment_capacity": [{"condition": ">", "enrollment": 25}, {"condition": "<", "enrollment": 30}],
                                }
                            },
                            id="sf7",
                        ),

                        # test section filters with current enrollment
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "enrollment": [{"condition": ">", "enrollment": 0}, {"condition": "<", "enrollment": 3}],
                                }
                            },
                            id="sf8",
                        ),

                        # test section filters with location
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "location": ["MAIN Campus, Surprenant Hall, Classroom, 134", "Online"],
                                }
                            },
                            id="sf9",
                        ),

                        # test section filters with meet times
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "meet_times": [
                                        {"day": "Monday", "start_time": "08:00", "end_time": "9:15"},
                                        {"day": "Tuesday", "start_time": "02:00", "end_time": "15:15"}
                                    ],
                                }
                            },
                            id="sf10",
                        ),

                        # test section filter with invalid filter keys (should be handled gracefully)
                        ToolCall(
                            name="section_filter",
                            args={
                                "filters": {
                                    "invalid_key": "invalid_value"
                                }
                            },
                            id="sf11",
                        ),

                        # test get student id by name with student assigned to user
                        ToolCall(name="get_student_id_by_name", args={"student_name": "John Doe"}, id="gsid1"),

                        # test get student id by name with student not assigned to user (should be handled gracefully)
                        ToolCall(name="get_student_id_by_name", args={"student_name": "Jane Smith"}, id="gsid2"),

                        # test get student id by name with nonexistent student (should be handled gracefully)
                        ToolCall(name="get_student_id_by_name", args={"student_name": "Nonexistent Student"}, id="gsid3"),

                        # test get advisor students
                        ToolCall(name="get_advisor_students", args={}, id="gas"),

                        # test get student basic info with student assigned to user
                        ToolCall(name="student_basic_info", args={"student_id": 324578}, id="sbi1"),

                        # test get student basic info with student not assigned to user (should be handled gracefully)
                        ToolCall(name="student_basic_info", args={"student_id": 987654}, id="sbi2"),

                        # test get student basic info with nonexistent student (should be handled gracefully)
                        ToolCall(name="student_basic_info", args={"student_id": 999999}, id="sbi3"),

                        # test get student course history with student assigned to user
                        ToolCall(name="student_course_history", args={"student_id": 324578}, id="sch1"),
                        
                        # test get student course history with student not assigned to user (should be handled gracefully)
                        ToolCall(name="student_course_history", args={"student_id": 987654}, id="sch2"),

                        # test get student course history with nonexistent student (should be handled gracefully)
                        ToolCall(name="student_course_history", args={"student_id": 999999}, id="sch3"),

                        # test get student interests with student assigned to user
                        ToolCall(name="student_interests", args={"student_id": 324578}, id="si1"),

                        # test get student interests with student not assigned to user (should be handled gracefully)
                        ToolCall(name="student_interests", args={"student_id": 987654}, id="si2"),

                        # test get student interests with nonexistent student (should be handled gracefully)
                        ToolCall(name="student_interests", args={"student_id": 999999}, id="si3"),

                        # test get student tracked sections with student assigned to user
                        ToolCall(name="student_tracked_sections", args={"student_id": 324578}, id="sts1"),

                        # test get student tracked sections with student not assigned to user (should be handled gracefully)
                        ToolCall(name="student_tracked_sections", args={"student_id": 987654}, id="sts2"),

                        # test get student tracked sections with nonexistent student (should be handled gracefully)
                        ToolCall(name="student_tracked_sections", args={"student_id": 999999}, id="sts3"),

                        # test get program requirements with real program
                        ToolCall(name="program_requirements", args={"program_name": "Computer Science Transfer"}, id="pr1"),

                        # test get program requirements with nonexistent program
                        ToolCall(name="program_requirements", args={"program_name": "Imaginary Program"}, id="pr2"),

                        # test get upcoming events
                        ToolCall(name="upcoming_events", args={}, id="ue"),

                        # test get event dates for real event
                        ToolCall(name="event_dates", args={"event_name": "AI Career Panel"}, id="ed1"),

                        # test get event dates for nonexistent event
                        ToolCall(name="event_dates", args={"event_name": "Nonexistent Event"}, id="ed2"),
                    ],
                ),
                AIMessage(content="Database tool smoke test complete: all tools were exercised and edge cases tested."),
                AIMessage(content="Database fallback: no additional database actions are needed."),
            ]
        )
    )

def _db_do_nothing_testing_model() -> FakeChatModel:
    """
    Create a DB helper test model that intentionally does nothing.

    This model returns simple messages with no `ToolCall`s. It's useful
    for verifying planner loop behavior when helpers do not request
    further actions (helps detect broken loop termination logic).

    Returns:
        FakeChatModel: a test model that produces no tool calls.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Called once."
                ),
                AIMessage(
                    content="Called twice."
                ),
                AIMessage(
                    content="Called thrice."
                ),
                AIMessage(
                    content="Called fourice."
                ),
                AIMessage(
                    content="Called fivice."
                ),
                AIMessage(
                    content="Should have stopped calling by now."
                )
            ],
        )
    )

def _db_loop_testing_model() -> FakeChatModel:
    """
    Create a DB helper model that issues tool calls across multiple loops.

    Produces a sequence of messages, each containing a `get_current_time`
    `ToolCall`. Used to validate loop limits are working properly.

    Returns:
        FakeChatModel: a loop-testing DB helper model.

    Notes: This should be used with db loop limit set to 3 for whichever version of the db agent you are testing.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Loop 1. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="get_current_time", args={}, id="ct1"),
                    ]
                ),
                AIMessage(
                    content="Loop 2. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="get_current_time", args={}, id="ct2"),
                    ]
                ),
                AIMessage(
                    content="Loop 3. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="get_current_time", args={}, id="ct3"),
                    ]
                ),
                AIMessage(
                    content="Loop 4. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="get_current_time", args={}, id="ct4"),
                    ]
                ),
                AIMessage(
                    content="Loop 5. Should have stopped by now. Limit may be broken. Ensure it is set to 3 for this test."
                )
            ]
        )
    )

def _web_tool_testing_model() -> FakeChatModel:
    """
    Create a fake student-web helper model that exercises web tools.

    The model produces `web_search` and `get_web_page_content` tool calls
    covering normal and edge-case inputs (empty queries, invalid URLs,
    large inputs). Use this to validate web helper behavior and error
    handling in tests.

    Returns:
        FakeChatModel: a test model instance for student web helper.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Comprehensive web-tool smoke test: exercise web_search and page fetch, including edge cases.",
                    tool_calls=[
                        # basic test web search
                        ToolCall(name="web_search", args={"query": "what is the weather today?"}, id="ws1"),

                        # test web search with max results
                        ToolCall(name="web_search", args={"query": "what is the weather today?", "max_results": 10}, id="ws2"),

                        # test web search with no results
                        ToolCall(name="web_search", args={"query": "jkhfjksdhgkjaergufv"}, id="ws3"),

                        # test web search with empty query
                        ToolCall(name="web_search", args={"query": ""}, id="ws4"),

                        # test web search with invalid max_results (should be handled gracefully)
                        ToolCall(name="web_search", args={"query": "what is the weather today?", "max_results": -5}, id="ws6"),

                        # test web search with query that triggers an error (e.g. by overwhelming the search engine or using disallowed content - should be handled gracefully)
                        ToolCall(name="web_search", args={"query": "a" * 1000}, id="ws7"),

                        # basic test page fetch
                        ToolCall(name="get_web_page_content", args={"url": "https://www.qcc.edu/"}, id="wp1"),

                        # test page fetch with max_chars
                        ToolCall(name="get_web_page_content", args={"url": "https://www.qcc.edu/", "max_chars": 100}, id="wp2"),

                        # test page fetch with url too many chars
                        ToolCall(name="get_web_page_content", args={"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"}, id="wp3"),

                        # test page fetch that is just a bean
                        ToolCall(name="get_web_page_content", args={"url": "https://justbean.co/"}, id="wp4"),

                        # test page fetch with url that redirects
                        ToolCall(name="get_web_page_content", args={"url": "http://github.com"}, id="wp5"),

                        # test page fetch with url that blocks bots
                        ToolCall(name="get_web_page_content", args={"url": "https://www.google.com/"}, id="wp6"),

                        # test page fetch with nonexistent page
                        ToolCall(name="get_web_page_content", args={"url": "https://www.fhskjgfjksd.com/"}, id="wp7"),

                        # test page fetch with invalid url
                        ToolCall(name="get_web_page_content", args={"url": "not a url"}, id="wp8"),
                    ],
                ),
                AIMessage(content="Web tool smoke test complete: search and fetch results collected and edge cases exercised."),
                AIMessage(content="Web fallback: no additional web searches are needed."),
            ]
        )
    )

def _web_do_nothing_testing_model() -> FakeChatModel:
    """
    Create a web helper test model that returns no actionable tool calls.

    Useful for testing planner loop termination and fallback behavior
    when the helper does not request further actions.

    Returns:
        FakeChatModel: a do-nothing web helper model.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Called once."
                ),
                AIMessage(
                    content="Called twice."
                ),
                AIMessage(
                    content="Called thrice."
                ),
                AIMessage(
                    content="Called fourice."
                ),
                AIMessage(
                    content="Called fivice."
                ),
                AIMessage(
                    content="Should have stopped calling by now."
                )
            ],
        )
    )

def _web_loop_testing_model() -> FakeChatModel:
    """
    Create a web helper model that issues web tool calls across loops.

    Produces several messages each requesting a `web_search`. Intended
    to validate loop limits and correct handling of repeated helper
    requests.

    Returns:
        FakeChatModel: a loop-testing web helper model.

    Note: This should be used with the web loop limit set to 3.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Loop 1. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="web_search", args={"query": "what is the weather today?"}, id="ws1"),
                    ]
                ),
                AIMessage(
                    content="Loop 2. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="web_search", args={"query": "what is the weather today?"}, id="ws2"),
                    ]
                ),
                AIMessage(
                    content="Loop 3. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="web_search", args={"query": "what is the weather today?"}, id="ws3"),
                    ]
                ),
                AIMessage(
                    content="Loop 4. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="web_search", args={"query": "what is the weather today?"}, id="ws4"),
                    ]
                ),
                AIMessage(
                    content="Loop 5. Should have stopped by now. Limit may be broken. Ensure it is set to 3 for this test."
                )
            ]
        )
    )

def _insertion_tool_testing_model() -> FakeChatModel:
    """
    Create a fake insertion-helper model that exercises insertion tools.

    The model generates `ToolCall`s for reading and writing student
    interests and tracked sections, including invalid inputs to test
    validation and error handling.

    Returns:
        FakeChatModel: a test model for insertion helper tooling.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Comprehensive insertion-tool smoke test: exercise read and write insertion tools and edge cases.",
                    tool_calls=[
                        # test get student interests
                        ToolCall(name="student_interests", args={}, id="gi"),

                        # test get student tracked sections
                        ToolCall(name="student_tracked_sections", args={}, id="gts1"),

                        # test insert student interests with valid interests
                        ToolCall(name="insert_student_interests", args={"interest": "Machine Learning"}, id="ii1"),

                        # test insert student interests with empty interest
                        ToolCall(name="insert_student_interests", args={"interest": ""}, id="ii2"),

                        # test insert tracked section with section not currently tracked
                        ToolCall(name="insert_student_tracked_sections", args={"course_code": "ACC 101", "section_id": "1"}, id="its1"),

                        # test insert tracked section with section already tracked (should be handled gracefully)
                        ToolCall(name="insert_student_tracked_sections", args={"course_code": "ACC 101", "section_id": "1"}, id="its2"),

                        # test insert tracked section with invalid section
                        ToolCall(name="insert_student_tracked_sections", args={"course_code": "ACC 101", "section_id": "99999"}, id="its3"),

                        # test insert tracked section with invalid section id type (should be handled gracefully)
                        ToolCall(name="insert_student_tracked_sections", args={"course_code": "ACC 101", "section_id": "not a section id"}, id="its4"),

                        # test insert tracked section with empty section_id
                        ToolCall(name="insert_student_tracked_sections", args={"course_code": "ACC 101", "section_id": ""}, id="its5"),

                        # check that the proper sections were tracked after the insertions
                        ToolCall(name="student_tracked_sections", args={}, id="gts2")
                    ],
                ),
                AIMessage(content="Insertion tool smoke test complete: reads and writes performed, including edge cases."),
                AIMessage(content="Insertion fallback: no additional insertions are needed."),
            ]
        )
    )

def _insertion_do_nothing_testing_model() -> FakeChatModel:
    """
    Create an insertion helper model that performs no insertions.

    Used to test planner loop termination and fallback when insertion
    helpers do not request further work.

    Returns:
        FakeChatModel: a do-nothing insertion helper model.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Called once."
                ),
                AIMessage(
                    content="Called twice."
                ),
                AIMessage(
                    content="Called thrice."
                ),
                AIMessage(
                    content="Called fourice."
                ),
                AIMessage(
                    content="Called fivice."
                ),
                AIMessage(
                    content="Should have stopped calling by now."
                )
            ]
        )
    )

def _insertion_loop_testing_model() -> FakeChatModel:
    """
    Create an insertion helper model that issues repeated tool calls.

    Produces several messages that each request `get_student_interests`.
    Helps validate loop counting and termination behavior for insertion
    helpers.

    Returns:
        FakeChatModel: a loop-testing insertion helper model.

    Note: This should be used with the insertion loop limit set to 3.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="Loop 1. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="get_student_interests", args={}, id="gi1"),
                    ]
                ),
                AIMessage(
                    content="Loop 2. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="get_student_interests", args={}, id="gi2"),
                    ]
                ),
                AIMessage(
                    content="Loop 3. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="get_student_interests", args={}, id="gi3"),
                    ]
                ),
                AIMessage(
                    content="Loop 4. Drag on to test loop limit.",
                    tool_calls=[
                        ToolCall(name="get_student_interests", args={}, id="gi4"),
                    ]
                ),
                AIMessage(
                    content="Loop 5. Should have stopped by now. Limit may be broken. Ensure it is set to 3 for this test."
                )
            ]
        )
    )

def _alerts_testing_model() -> FakeChatModel:
    """
    Create a simple alerts agent test model.

    Produces messages with serialized JSON containing `relivent_events`
    (sic) lists to simulate alerts being detected or none found. Used in
    tests for alerts processing logic.

    Returns:
        FakeChatModel: a test model for the alerts agent.
    """
    return FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=json.dumps(
                        {
                            "relivent_events": [
                                {"ID": 12, "Urgency": 5},
                                {"ID": 27, "Urgency": 3},
                            ]
                        }
                    )
                ),
                AIMessage(
                    content=json.dumps(
                        {
                            "relivent_events": [],
                        }
                    )
                ),
            ]
        )
    )

def _create_model(model_name: str, node_name: str):
    """
    Factory to create a chat model instance for a given node.

    Args:
        model_name (str): The model name from configuration (e.g. "sonnet-4-6",
            "gpt_4o", or one of the special testing identifiers like
            "s_tool_test", "a_tool_test", "loop_test", "do_nothing_test").
        node_name (str): The node type requesting the model ("planning",
            "db", "web", "insertion", "alerts").

    Returns:
        An instance of a chat model (langchain Chat model or
        `FakeChatModel`) appropriate for the request.

    Raises:
        ValueError: if required environment API keys are missing or if an
            invalid model or node name is provided.
    """
    normalized = _normalize_model_name(model_name)

    match normalized:
        case "sonnet_4_6":
            if env := os.getenv("ANTHROPIC_API_KEY"):
                return ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2)
            else:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        case "gpt_4o":
            if env := os.getenv("OPENAI_API_KEY"):
                return ChatOpenAI(model="gpt-4o", temperature=0.2)
            else:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
        case "free":
            if env := os.getenv("OPENROUTER_API_KEY"):
                return ChatOpenRouter(model="openrouter/free", temperature=0.2)
            else:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
        case "ollama" | "qwen3_14b":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
            return ChatOllama(
                model="qwen3:14b",
                base_url=base_url,
                temperature=0.2,
            )
        case "s_tool_test":
            match node_name:
                case "planning":
                    return _s_planner_tool_testing_model()
                case "db":
                    return _s_db_tool_testing_model()
                case _:
                    raise ValueError(f"Invalid node name {node_name} for model {model_name}")
        case "a_tool_test":
            match node_name:
                case "planning":
                    return _a_planner_tool_testing_model()
                case "db":
                    return _a_db_tool_testing_model()
                case _:
                    raise ValueError(f"Invalid node name {node_name} for model {model_name}")
        case "tool_test":
            match node_name:
                case "web":
                    return _web_tool_testing_model()
                case "insertion":
                    return _insertion_tool_testing_model()
                case _:
                    raise ValueError(f"Invalid node name {node_name} for model {model_name}")
        case "loop_test":
            match node_name:
                case "planning":
                    return _planner_loop_testing_model()
                case "db":
                    return _db_loop_testing_model()
                case "web":
                    return _web_loop_testing_model()
                case "insertion":
                    return _insertion_loop_testing_model()
                case _:
                    raise ValueError(f"Invalid node name {node_name} for model {model_name}")
        case "do_nothing_test":
            match node_name:
                case "db":
                    return _db_do_nothing_testing_model()
                case "web":
                    return _web_do_nothing_testing_model()
                case "insertion":
                    return _insertion_do_nothing_testing_model()
                case _:
                    raise ValueError(f"Invalid node name {node_name} for model {model_name}")
        case "test":
            match node_name:
                case "alerts":
                    return _alerts_testing_model()
                case _:
                    raise ValueError(f"Invalid node name {node_name} for model {model_name}")
        case _:
            raise ValueError(f"Invalid model name {model_name}")

model_select, mode = _load_model_config()

planning_llm = _create_model(model_select["planning"], "planning")
db_llm = _create_model(model_select["db"], "db")
web_llm = _create_model(model_select["web"], "web")
insertion_llm = _create_model(model_select["insertion"], "insertion")
alerts_llm = _create_model(model_select["alerts"], "alerts")
