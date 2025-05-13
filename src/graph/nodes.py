# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents.agents import coder_agent, research_agent, create_agent

from src.tools.search import LoggedTavilySearch
from src.tools import (
    crawl_tool,
    web_search_tool,
    python_repl_tool,
)

from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.planner_model import Plan, StepType
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from .types import State
from ..config import SEARCH_MAX_RESULTS, SELECTED_SEARCH_ENGINE, SearchEngine

# Configure enhanced logging
logger = logging.getLogger(__name__)

def log_node_entry(node_name, state=None):
    """Log entry to a node with basic state info"""
    logger.info(f"ENTERING NODE: {node_name}")
    if state:
        logger.debug(f"{node_name} - Input state keys: {list(state.keys())}")
        if "messages" in state and state["messages"]:
            last_message = state["messages"][-1] if state["messages"] else None
            logger.debug(f"{node_name} - Last message: {last_message}")

def log_node_exit(node_name, result=None):
    """Log exit from a node with result info"""
    logger.info(f"EXITING NODE: {node_name}")
    if result:
        if isinstance(result, dict):
            logger.debug(f"{node_name} - Result keys: {list(result.keys())}")
        elif hasattr(result, "goto"):
            logger.info(f"{node_name} - Next node: {result.goto}")
            if hasattr(result, "update") and result.update:
                logger.debug(f"{node_name} - Update keys: {list(result.update.keys())}")


@tool
def handoff_to_planner(
    task_title: Annotated[str, "The title of the task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    logger.debug(f"Tool called: handoff_to_planner with title={task_title}, locale={locale}")
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return


def background_investigation_node(state: State) -> Command[Literal["planner"]]:
    log_node_entry("background_investigation", state)
    logger.info("Background investigation started for query")
    
    query = state["messages"][-1].content
    logger.info(f"Search query: {query}")
    
    try:
        if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY:
            logger.info(f"Using Tavily search with max_results={SEARCH_MAX_RESULTS}")
            searched_content = LoggedTavilySearch(max_results=SEARCH_MAX_RESULTS).invoke(
                {"query": query}
            )
            background_investigation_results = None
            if isinstance(searched_content, list):
                background_investigation_results = [
                    {"title": elem["title"], "content": elem["content"]}
                    for elem in searched_content
                ]
                logger.info(f"Found {len(background_investigation_results)} results from Tavily")
                logger.debug(f"Search results: {json.dumps(background_investigation_results, indent=2)[:500]}...")
            else:
                logger.error(
                    f"Tavily search returned malformed response: {searched_content}"
                )
        else:
            logger.info(f"Using {SELECTED_SEARCH_ENGINE} search")
            background_investigation_results = web_search_tool.invoke(query)
            logger.info(f"Web search completed with results")
            logger.debug(f"Web search results: {json.dumps(background_investigation_results, indent=2)[:500]}...")
    except Exception as e:
        logger.error(f"Search failed with error: {str(e)}")
        import traceback
        logger.debug(f"Search error details: {traceback.format_exc()}")
        background_investigation_results = []
    
    result = Command(
        update={
            "background_investigation_results": json.dumps(
                background_investigation_results, ensure_ascii=False
            )
        },
        goto="planner",
    )
    
    log_node_exit("background_investigation", result)
    return result


def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["human_feedback", "reporter"]]:
    """Planner node that generate the full plan."""
    log_node_entry("planner", state)
    logger.info("Planner generating full plan")
    
    try:
        configurable = Configuration.from_runnable_config(config)
        plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
        logger.info(f"Plan iteration: {plan_iterations}/{configurable.max_plan_iterations}")
        
        # Apply prompt template
        messages = apply_prompt_template("planner", state, configurable)
        logger.debug(f"Planner prompt template applied, messages count: {len(messages)}")

        # Add background investigation results if available
        if (
            plan_iterations == 0
            and state.get("enable_background_investigation")
            and state.get("background_investigation_results")
        ):
            logger.info("Adding background investigation results to planner input")
            messages += [
                {
                    "role": "user",
                    "content": (
                        "background investigation results of user query:\n"
                        + state["background_investigation_results"]
                        + "\n"
                    ),
                }
            ]

        # Configure LLM
        if AGENT_LLM_MAP["planner"] == "basic":
            logger.info("Using basic LLM with structured output for planner")
            llm = get_llm_by_type(AGENT_LLM_MAP["planner"]).with_structured_output(
                Plan,
                method="json_mode",
            )
        else:
            logger.info(f"Using {AGENT_LLM_MAP['planner']} LLM for planner")
            llm = get_llm_by_type(AGENT_LLM_MAP["planner"])

        # Check plan iterations limit
        if plan_iterations >= configurable.max_plan_iterations:
            logger.info(f"Reached max plan iterations ({configurable.max_plan_iterations}), going to reporter")
            result = Command(goto="reporter")
            log_node_exit("planner", result)
            return result

        # Invoke LLM
        full_response = ""
        logger.info("Invoking LLM for planning")
        if AGENT_LLM_MAP["planner"] == "basic":
            response = llm.invoke(messages)
            full_response = response.model_dump_json(indent=4, exclude_none=True)
            logger.debug("Received structured JSON response from planner LLM")
        else:
            response = llm.stream(messages)
            for chunk in response:
                full_response += chunk.content
            logger.debug("Received streamed response from planner LLM")
            
        logger.info("Planner LLM response received")
        logger.debug(f"Planner response: {full_response[:500]}...")

        # Parse JSON
        try:
            curr_plan = json.loads(repair_json_output(full_response))
            logger.info("Successfully parsed planner JSON response")
            logger.debug(f"Parsed plan: {json.dumps(curr_plan, indent=2)[:500]}...")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse planner response as JSON: {str(e)}")
            logger.debug(f"Invalid JSON: {full_response}")
            if plan_iterations > 0:
                result = Command(goto="reporter")
            else:
                result = Command(goto="__end__")
            log_node_exit("planner", result)
            return result
            
        # Check if plan has enough context
        if curr_plan.get("has_enough_context"):
            logger.info("Plan has enough context, validating as Plan model")
            try:
                new_plan = Plan.model_validate(curr_plan)
                logger.info(f"Plan validated successfully with {len(new_plan.steps)} steps")
                result = Command(
                    update={
                        "messages": [AIMessage(content=full_response, name="planner")],
                        "current_plan": new_plan,
                    },
                    goto="reporter",
                )
            except Exception as e:
                logger.error(f"Plan validation failed: {str(e)}")
                result = Command(
                    update={
                        "messages": [AIMessage(content=full_response, name="planner")],
                        "current_plan": full_response,
                    },
                    goto="human_feedback",
                )
        else:
            logger.info("Plan needs more context, going to human feedback")
            result = Command(
                update={
                    "messages": [AIMessage(content=full_response, name="planner")],
                    "current_plan": full_response,
                },
                goto="human_feedback",
            )
    except Exception as e:
        logger.error(f"Error in planner node: {str(e)}")
        import traceback
        logger.debug(f"Planner error details: {traceback.format_exc()}")
        result = Command(goto="__end__")
        
    log_node_exit("planner", result)
    return result


def human_feedback_node(
    state,
) -> Command[Literal["planner", "research_team", "reporter", "__end__"]]:
    log_node_entry("human_feedback", state)
    
    try:
        current_plan = state.get("current_plan", "")
        logger.info("Processing human feedback for plan")
        
        # Check if plan is auto-accepted
        auto_accepted_plan = state.get("auto_accepted_plan", False)
        if not auto_accepted_plan:
            logger.info("Plan requires explicit user approval, interrupting for feedback")
            feedback = interrupt("Please Review the Plan.")
            logger.info(f"Received feedback: {feedback}")

            # Process feedback
            if feedback and str(feedback).upper().startswith("[EDIT_PLAN]"):
                logger.info("User requested plan edit, returning to planner")
                result = Command(
                    update={
                        "messages": [
                            HumanMessage(content=feedback, name="feedback"),
                        ],
                    },
                    goto="planner",
                )
                log_node_exit("human_feedback", result)
                return result
            elif feedback and str(feedback).upper().startswith("[ACCEPTED]"):
                logger.info("Plan accepted by user")
            else:
                logger.error(f"Unsupported feedback format: {feedback}")
                raise TypeError(f"Interrupt value of {feedback} is not supported.")
        else:
            logger.info("Auto-accepted plan, no user feedback required")

        # Process accepted plan
        plan_iterations = state.get("plan_iterations", 0)
        logger.info(f"Current plan iteration: {plan_iterations}")
        goto = "research_team"
        
        try:
            logger.debug("Repairing and parsing plan JSON")
            current_plan = repair_json_output(current_plan)
            new_plan = json.loads(current_plan)
            
            # Increment plan iterations
            plan_iterations += 1
            logger.info(f"Incrementing plan iterations to {plan_iterations}")
            
            # Check if plan has enough context
            if new_plan["has_enough_context"]:
                logger.info("Plan has enough context, going to reporter")
                goto = "reporter"
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {str(e)}")
            if plan_iterations > 0:
                logger.info("Plan iterations > 0, going to reporter despite JSON error")
                result = Command(goto="reporter")
            else:
                logger.info("Plan iterations = 0, ending workflow due to JSON error")
                result = Command(goto="__end__")
            log_node_exit("human_feedback", result)
            return result

        # Create plan model and return
        logger.info(f"Proceeding to {goto} with validated plan")
        try:
            validated_plan = Plan.model_validate(new_plan)
            logger.info(f"Plan validated with {len(validated_plan.steps)} steps")
            result = Command(
                update={
                    "current_plan": validated_plan,
                    "plan_iterations": plan_iterations,
                    "locale": new_plan["locale"],
                },
                goto=goto,
            )
        except Exception as e:
            logger.error(f"Plan validation failed: {str(e)}")
            result = Command(goto="__end__")
    except Exception as e:
        logger.error(f"Error in human feedback node: {str(e)}")
        import traceback
        logger.debug(f"Human feedback error details: {traceback.format_exc()}")
        result = Command(goto="__end__")
    
    log_node_exit("human_feedback", result)
    return result


def coordinator_node(
    state: State,
) -> Command[Literal["planner", "background_investigator", "__end__"]]:
    """Coordinator node that communicate with customers."""
    log_node_entry("coordinator", state)
    logger.info("Coordinator processing user input")
    
    try:
        # Apply prompt template
        messages = apply_prompt_template("coordinator", state)
        logger.debug(f"Coordinator prompt applied with {len(messages)} messages")
        
        # Invoke LLM with tools
        logger.info(f"Invoking {AGENT_LLM_MAP['coordinator']} LLM with handoff_to_planner tool")
        response = (
            get_llm_by_type(AGENT_LLM_MAP["coordinator"])
            .bind_tools([handoff_to_planner])
            .invoke(messages)
        )
        logger.debug(f"Coordinator response received: {response}")

        # Initialize defaults
        goto = "__end__"
        locale = state.get("locale", "en-US")
        logger.info(f"Initial locale: {locale}")

        # Process tool calls
        if response.tool_calls:
            logger.info(f"Found {len(response.tool_calls)} tool calls in response")
            goto = "planner"
            if state.get("enable_background_investigation"):
                logger.info("Background investigation enabled, adjusting path")
                goto = "background_investigator"
                
            try:
                for tool_call in response.tool_calls:
                    logger.debug(f"Processing tool call: {tool_call}")
                    if tool_call.get("name", "") != "handoff_to_planner":
                        logger.warning(f"Unexpected tool call: {tool_call.get('name')}")
                        continue
                        
                    if tool_locale := tool_call.get("args", {}).get("locale"):
                        logger.info(f"Setting locale from tool call: {tool_locale}")
                        locale = tool_locale
                        break
            except Exception as e:
                logger.error(f"Error processing tool calls: {e}")
                import traceback
                logger.debug(f"Tool call processing error: {traceback.format_exc()}")
        else:
            logger.warning("No tool calls found in coordinator response, ending workflow")
            logger.debug(f"Coordinator response content: {response.content}")

        # Return command
        result = Command(
            update={"locale": locale},
            goto=goto,
        )
    except Exception as e:
        logger.error(f"Error in coordinator node: {str(e)}")
        import traceback
        logger.debug(f"Coordinator error details: {traceback.format_exc()}")
        result = Command(goto="__end__")
    
    log_node_exit("coordinator", result)
    return result


def reporter_node(state: State):
    """Reporter node that write a final report."""
    log_node_entry("reporter", state)
    logger.info("Reporter generating final report")
    
    try:
        # Get current plan
        current_plan = state.get("current_plan")
        if not current_plan:
            logger.warning("No current plan available for reporter")
            return {"final_report": "No plan available to generate report."}
            
        logger.info(f"Generating report for plan: {current_plan.title}")
        
        # Prepare input
        input_ = {
            "messages": [
                HumanMessage(
                    f"# Research Requirements\n\n## Task\n\n{current_plan.title}\n\n## Description\n\n{current_plan.thought}"
                )
            ],
            "locale": state.get("locale", "en-US"),
        }
        logger.debug(f"Reporter input prepared with locale: {input_['locale']}")
        
        # Apply prompt template
        invoke_messages = apply_prompt_template("reporter", input_)
        logger.debug(f"Reporter prompt template applied with {len(invoke_messages)} messages")
        
        # Add observations
        observations = state.get("observations", [])
        logger.info(f"Adding {len(observations)} observations to reporter input")

        # Add report format guidance
        invoke_messages.append(
            HumanMessage(
                content="IMPORTANT: Structure your report according to the format in the prompt. Remember to include:\n\n1. Key Points - A bulleted list of the most important findings\n2. Overview - A brief introduction to the topic\n3. Detailed Analysis - Organized into logical sections\n4. Survey Note (optional) - For more comprehensive reports\n5. Key Citations - List all references at the end\n\nFor citations, DO NOT include inline citations in the text. Instead, place all citations in the 'Key Citations' section at the end using the format: `- [Source Title](URL)`. Include an empty line between each citation for better readability.\n\nPRIORITIZE USING MARKDOWN TABLES for data presentation and comparison. Use tables whenever presenting comparative data, statistics, features, or options. Structure tables with clear headers and aligned columns. Example table format:\n\n| Feature | Description | Pros | Cons |\n|---------|-------------|------|------|\n| Feature 1 | Description 1 | Pros 1 | Cons 1 |\n| Feature 2 | Description 2 | Pros 2 | Cons 2 |",
                name="system",
            )
        )

        # Add observation messages
        for i, observation in enumerate(observations):
            logger.debug(f"Adding observation {i+1} to reporter")
            invoke_messages.append(
                HumanMessage(
                    content=f"Below are some observations for the research task:\n\n{observation}",
                    name="observation",
                )
            )
            
        # Invoke reporter LLM
        logger.info(f"Invoking {AGENT_LLM_MAP['reporter']} LLM for report generation")
        response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(invoke_messages)
        response_content = response.content
        
        # Log and return result
        logger.info("Report generated successfully")
        logger.debug(f"Report content length: {len(response_content)} characters")
        logger.debug(f"Report preview: {response_content[:500]}...")
        
        result = {"final_report": response_content}
    except Exception as e:
        logger.error(f"Error in reporter node: {str(e)}")
        import traceback
        logger.debug(f"Reporter error details: {traceback.format_exc()}")
        result = {"final_report": f"Error generating report: {str(e)}"}
    
    log_node_exit("reporter", result)
    return result


def research_team_node(
    state: State,
) -> Command[Literal["planner", "researcher", "coder"]]:
    """Research team node that collaborates on tasks."""
    log_node_entry("research_team", state)
    logger.info("Research team determining next task")
    
    try:
        # Check current plan
        current_plan = state.get("current_plan")
        if not current_plan or not current_plan.steps:
            logger.warning("No plan or empty steps, returning to planner")
            result = Command(goto="planner")
            log_node_exit("research_team", result)
            return result
            
        # Check if all steps are executed
        if all(step.execution_res for step in current_plan.steps):
            logger.info("All steps executed, returning to planner")
            result = Command(goto="planner")
            log_node_exit("research_team", result)
            return result
            
        # Find next unexecuted step
        for step in current_plan.steps:
            if not step.execution_res:
                logger.info(f"Found unexecuted step: {step.title}")
                break
                
        # Determine next agent based on step type
        if step.step_type and step.step_type == StepType.RESEARCH:
            logger.info("Research step identified, going to researcher")
            result = Command(goto="researcher")
        elif step.step_type and step.step_type == StepType.PROCESSING:
            logger.info("Processing step identified, going to coder")
            result = Command(goto="coder")
        else:
            logger.warning(f"Unknown step type: {step.step_type}, returning to planner")
            result = Command(goto="planner")
    except Exception as e:
        logger.error(f"Error in research team node: {str(e)}")
        import traceback
        logger.debug(f"Research team error details: {traceback.format_exc()}")
        result = Command(goto="planner")
    
    log_node_exit("research_team", result)
    return result


async def _execute_agent_step(
    state: State, agent, agent_name: str
) -> Command[Literal["research_team"]]:
    """Helper function to execute a step using the specified agent."""
    log_node_entry(f"{agent_name}_step_execution", state)
    logger.info(f"Executing step with {agent_name} agent")
    
    try:
        current_plan = state.get("current_plan")
        observations = state.get("observations", [])

        # Find the first unexecuted step
        for step in current_plan.steps:
            if not step.execution_res:
                break

        logger.info(f"Executing step: {step.title}")
        logger.debug(f"Step description: {step.description[:200]}...")

        # Prepare the input for the agent
        agent_input = {
            "messages": [
                HumanMessage(
                    content=f"#Task\n\n##title\n\n{step.title}\n\n##description\n\n{step.description}\n\n##locale\n\n{state.get('locale', 'en-US')}"
                )
            ]
        }
        logger.debug(f"Agent input prepared for {agent_name}")

        # Add citation reminder for researcher agent
        if agent_name == "researcher":
            logger.debug("Adding citation reminder for researcher agent")
            agent_input["messages"].append(
                HumanMessage(
                    content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
                    name="system",
                )
            )

        # Invoke the agent
        logger.info(f"Invoking {agent_name} agent")
        result = await agent.ainvoke(input=agent_input)
        logger.info(f"{agent_name} agent execution completed")

        # Process the result
        if not result or "messages" not in result or not result["messages"]:
            logger.warning(f"{agent_name} agent returned empty result")
            response_content = f"Error: {agent_name} agent returned no results"
        else:
            response_content = result["messages"][-1].content
            logger.debug(f"{agent_name} response content length: {len(response_content)} characters")
            logger.debug(f"{agent_name} response preview: {response_content[:500]}...")

        # Update the step with the execution result
        step.execution_res = response_content
        logger.info(f"Step '{step.title}' execution result saved")

        # Prepare command
        result = Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response_content,
                        name=agent_name,
                    )
                ],
                "observations": observations + [response_content],
            },
            goto="research_team",
        )
    except Exception as e:
        logger.error(f"Error executing {agent_name} step: {str(e)}")
        import traceback
        logger.debug(f"{agent_name} execution error details: {traceback.format_exc()}")
        
        # Create error content and update step
        error_content = f"Error executing {agent_name} step: {str(e)}"
        if "step" in locals() and step:
            step.execution_res = error_content
            
        # Return command with error
        result = Command(
            update={
                "messages": [
                    HumanMessage(
                        content=error_content,
                        name=agent_name,
                    )
                ],
                "observations": observations + [error_content] if "observations" in locals() else [error_content],
            },
            goto="research_team",
        )
    
    log_node_exit(f"{agent_name}_step_execution", result)
    return result


async def _setup_and_execute_agent_step(
    state: State,
    config: RunnableConfig,
    agent_type: str,
    default_agent,
    default_tools: list,
) -> Command[Literal["research_team"]]:
    """Helper function to set up an agent with appropriate tools and execute a step."""
    log_node_entry(f"{agent_type}_setup", state)
    logger.info(f"Setting up {agent_type} agent with tools")
    
    try:
        configurable = Configuration.from_runnable_config(config)
        mcp_servers = {}
        enabled_tools = {}

        # Extract MCP server configuration
        if configurable.mcp_settings:
            logger.info("Processing MCP settings for agent")
            for server_name, server_config in configurable.mcp_settings["servers"].items():
                if (
                    server_config["enabled_tools"]
                    and agent_type in server_config["add_to_agents"]
                ):
                    logger.info(f"Adding MCP server {server_name} for {agent_type}")
                    mcp_servers[server_name] = {
                        k: v
                        for k, v in server_config.items()
                        if k in ("transport", "command", "args", "url", "env")
                    }
                    for tool_name in server_config["enabled_tools"]:
                        enabled_tools[tool_name] = server_name
                        logger.debug(f"Enabled tool {tool_name} from {server_name}")

        # Create and execute agent with MCP tools if available
        if mcp_servers:
            logger.info(f"Using MCP client with {len(mcp_servers)} servers for {agent_type}")
            async with MultiServerMCPClient(mcp_servers) as client:
                loaded_tools = default_tools[:]
                logger.debug(f"Loaded {len(loaded_tools)} default tools")
                
                client_tools = client.get_tools()
                logger.debug(f"MCP client provided {len(client_tools)} tools")
                
                for tool in client_tools:
                    if tool.name in enabled_tools:
                        tool.description = (
                            f"Powered by '{enabled_tools[tool.name]}'.\n{tool.description}"
                        )
                        loaded_tools.append(tool)
                        logger.debug(f"Added MCP tool: {tool.name}")
                
                logger.info(f"Creating {agent_type} agent with {len(loaded_tools)} tools")
                agent = create_agent(agent_type, agent_type, loaded_tools, agent_type)
                return await _execute_agent_step(state, agent, agent_type)
        else:
            # Use default agent
            logger.info(f"No MCP servers configured, using default {agent_type} agent")
            return await _execute_agent_step(state, default_agent, agent_type)
    except Exception as e:
        logger.error(f"Error setting up {agent_type} agent: {str(e)}")
        import traceback
        logger.debug(f"{agent_type} setup error details: {traceback.format_exc()}")
        
        # Return error command
        observations = state.get("observations", [])
        error_content = f"Error setting up {agent_type} agent: {str(e)}"
        result = Command(
            update={
                "messages": [
                    HumanMessage(
                        content=error_content,
                        name=agent_type,
                    )
                ],
                "observations": observations + [error_content],
            },
            goto="research_team",
        )
        
        log_node_exit(f"{agent_type}_setup", result)
        return result


async def researcher_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node that do research"""
    log_node_entry("researcher", state)
    logger.info("Researcher node processing research task")
    
    result = await _setup_and_execute_agent_step(
        state,
        config,
        "researcher",
        research_agent,
        [web_search_tool, crawl_tool],
    )
    
    log_node_exit("researcher", result)
    return result


async def coder_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Coder node that do code analysis."""
    log_node_entry("coder", state)
    logger.info("Coder node processing coding task")
    
    result = await _setup_and_execute_agent_step(
        state,
        config,
        "coder",
        coder_agent,
        [python_repl_tool],
    )
    
    log_node_exit("coder", result)
    return result
