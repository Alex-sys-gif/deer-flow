# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import logging
import json
import traceback
import os
from logging.handlers import RotatingFileHandler
from src.graph import build_graph

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Setup logging to file and console
log_file = os.path.join(log_dir, "deerflow.log")
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        file_handler,
        logging.StreamHandler()  # Console output
    ]
)


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# Create the graph
graph = build_graph()


async def run_agent_workflow_async(
    user_input: str,
    debug: bool = False,
    max_plan_iterations: int = 1,
    max_step_num: int = 3,
    enable_background_investigation: bool = True,
):
    """Run the agent workflow asynchronously with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging
        max_plan_iterations: Maximum number of plan iterations
        max_step_num: Maximum number of steps in a plan
        enable_background_investigation: If True, performs web search before planning to enhance context

    Returns:
        The final state after the workflow completes
    """
    if not user_input:
        logger.error("Empty user input provided")
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()
        logger.debug("Debug logging enabled")

    logger.info(f"Starting async workflow with user input: {user_input}")
    logger.info(f"Configuration: max_plan_iterations={max_plan_iterations}, max_step_num={max_step_num}, enable_background_investigation={enable_background_investigation}")
    
    initial_state = {
        # Runtime Variables
        "messages": [{"role": "user", "content": user_input}],
        "auto_accepted_plan": True,
        "enable_background_investigation": enable_background_investigation,
    }
    logger.info(f"Initial state prepared: {json.dumps(initial_state, indent=2)}")
    
    config = {
        "configurable": {
            "thread_id": "default",
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            "mcp_settings": {
                "servers": {
                    "mcp-github-trending": {
                        "transport": "stdio",
                        "command": "uvx",
                        "args": ["mcp-github-trending"],
                        "enabled_tools": ["get_github_trending_repositories"],
                        "add_to_agents": ["researcher"],
                    }
                }
            },
        },
        "recursion_limit": 100,
    }
    logger.info("Starting graph execution with configuration")
    logger.debug(f"Full configuration: {json.dumps(config, indent=2)}")
    
    last_message_cnt = 0
    final_state = None
    
    try:
        async for s in graph.astream(
            input=initial_state, config=config, stream_mode="values"
        ):
            try:
                logger.debug(f"Received state update: {type(s)}")
                
                if isinstance(s, dict):
                    final_state = s  # Keep track of final state
                    
                    if "messages" in s:
                        if len(s["messages"]) <= last_message_cnt:
                            logger.debug("No new messages in this update, skipping")
                            continue
                            
                        new_message_count = len(s["messages"]) - last_message_cnt
                        logger.info(f"Processing {new_message_count} new message(s)")
                        last_message_cnt = len(s["messages"])
                        
                        message = s["messages"][-1]
                        logger.debug(f"Latest message type: {type(message)}")
                        
                        if message is None:
                            logger.warning("Received None message in state update")
                        
                        if isinstance(message, tuple):
                            logger.info(f"Tuple message: {message}")
                            print(message)
                        else:
                            logger.debug(f"Message content: {getattr(message, 'content', message)}")
                            message.pretty_print()
                    else:
                        logger.debug(f"State update missing 'messages' key: {list(s.keys())}")
                else:
                    # For any other output format
                    logger.info(f"Non-dict output type: {type(s)}")
                    print(f"Output: {s}")
                    
            except Exception as e:
                logger.error(f"Error processing stream output: {e}")
                logger.debug(f"Error details: {traceback.format_exc()}")
                print(f"Error processing output: {str(e)}")
    except Exception as e:
        logger.error(f"Critical error in workflow execution: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        raise

    # Log final state information
    if final_state:
        logger.info("Workflow completed with final state")
        if "messages" in final_state:
            logger.info(f"Final message count: {len(final_state['messages'])}")
            last_message = final_state["messages"][-1] if final_state["messages"] else None
            if last_message is None:
                logger.warning("Final message is None - this may cause empty UI results")
            else:
                logger.info(f"Final message type: {type(last_message)}")
                logger.debug(f"Final message content: {getattr(last_message, 'content', last_message)}")
    else:
        logger.warning("Workflow completed without final state")

    logger.info("Async workflow completed successfully")
    return final_state


if __name__ == "__main__":
    print(graph.get_graph(xray=True).draw_mermaid())
