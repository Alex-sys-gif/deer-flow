# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import json
import json_repair
import re

logger = logging.getLogger(__name__)


def repair_json_output(content: str) -> str:
    """
    Repair and normalize JSON output.

    Args:
        content (str): String content that may contain JSON

    Returns:
        str: Repaired JSON string, or original content if not JSON
    """
    content = content.strip()
    if content.startswith(("{", "[")) or "```json" in content or "```ts" in content:
        try:
            # If content is wrapped in ```json code block, extract the JSON part
            if content.startswith("```json"):
                content = content.removeprefix("```json")

            if content.startswith("```ts"):
                content = content.removeprefix("```ts")

            if content.endswith("```"):
                content = content.removesuffix("```")
                
            # Additional fixes before using json_repair
            # Fix missing commas between properties
            content = re.sub(r'"\s*(?="\w+")', '", ', content)
            
            # Fix missing commas after objects in arrays
            content = re.sub(r'}\s*{', '}, {', content)
            
            # Add missing commas after arrays
            content = re.sub(r']\s*{', '], {', content)
            content = re.sub(r'}\s*\[', '}, [', content)
            
            # Balance brackets if needed
            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces > close_braces:
                content += '}' * (open_braces - close_braces)
                
            open_brackets = content.count('[')
            close_brackets = content.count(']')
            if open_brackets > close_brackets:
                content += ']' * (open_brackets - close_brackets)
            
            logger.debug(f"Pre-processed JSON content: {content}")

            # Try to repair and parse JSON
            repaired_content = json_repair.loads(content)
            return json.dumps(repaired_content, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}")
            logger.debug(f"Failed JSON content: {content}")
    return content
