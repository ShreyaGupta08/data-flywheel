# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tool extraction and normalization utilities for processing model inference outputs.
Handles various tool call formats and normalizes them to standard OpenAI format.
"""

import json
import re
from pathlib import Path
from typing import Any

from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.tool_normalizer")


def extract_tool_info(record: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """
    Extract tool name and arguments from various formats.

    Handles multiple formats:
    - Standard OpenAI tool_calls format
    - JSON in content field
    - XML-like format in content field
    - Tool names mentioned in content

    Args:
        record: A dictionary containing 'content' or 'tool_calls' keys

    Returns:
        Tuple of (tool_name, arguments_dict) or (None, None) if not found.
    """
    content = record.get("content")
    tool_calls = record.get("tool_calls")

    # Case 1: Already has tool_calls array
    if tool_calls and len(tool_calls) > 0:
        tc = tool_calls[0]
        func = tc.get("function", {})
        name = func.get("name")
        args = func.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"query": args}
        return name, args or {}

    if not content:
        return None, None

    # Clean up content - remove common prefixes
    clean_content = content.strip()
    clean_content = re.sub(r"^<\|python_tag\|>", "", clean_content)
    clean_content = re.sub(r"^function=\w+", "", clean_content)
    clean_content = re.sub(r"^\w+=", "", clean_content)

    # Case 2: Try to parse as JSON
    try:
        json_match = re.search(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*\}', clean_content)
        if json_match:
            try:
                data = json.loads(
                    clean_content.split("}")[0] + "}" if "}" in clean_content else clean_content
                )
            except json.JSONDecodeError:
                brace_count = 0
                start = clean_content.find("{")
                data = None
                if start != -1:
                    for i, c in enumerate(clean_content[start:], start):
                        if c == "{":
                            brace_count += 1
                        elif c == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                try:
                                    data = json.loads(clean_content[start : i + 1])
                                    break
                                except json.JSONDecodeError:
                                    pass

            if data and isinstance(data, dict):
                name = data.get("name")
                params = data.get("parameters", {})
                if isinstance(params, dict):
                    if "properties" in params:
                        props = params["properties"]
                        if isinstance(props, dict):
                            return name, props
                        elif isinstance(props, str):
                            return name, {"query": props}
                    return name, params
                return name, {"query": str(params)}
    except Exception:
        pass

    # Case 3: XML-like format
    xml_patterns = [
        r"<(ToOrderStatusAssistant|ToReturnProcessing|ToProductQAAssistant|HandleOtherTalk)\s+([^>]+?)/>",
        r"<(ToOrderStatusAssistant|ToReturnProcessing|ToProductQAAssistant|HandleOtherTalk)\s+([^>]+?)>.*?</\1>",
        r"<(ToOrderStatusAssistant|ToReturnProcessing|ToProductQAAssistant|HandleOtherTalk)\s+([^>]+?)>",
    ]

    for pattern in xml_patterns:
        xml_match = re.search(pattern, content, re.DOTALL)
        if xml_match:
            name = xml_match.group(1)
            attrs_str = xml_match.group(2)
            args: dict[str, Any] = {}
            for attr_match in re.finditer(r"(\w+)=[\"']?([^\"'>\s]+)[\"']?", attrs_str):
                args[attr_match.group(1)] = attr_match.group(2)
            for attr_match in re.finditer(r'(\w+)="([^"]*)"', attrs_str):
                args[attr_match.group(1)] = attr_match.group(2)
            return name, args

    # Case 4: Check for tool names mentioned in content
    tool_names = [
        "HandleOtherTalk",
        "ToProductQAAssistant",
        "ToOrderStatusAssistant",
        "ToReturnProcessing",
    ]
    for tool_name in tool_names:
        if tool_name in content:
            query_match = re.search(r'"query"\s*:\s*"([^"]+)"', content)
            user_id_match = re.search(r'"user_id"\s*:\s*"([^"]+)"', content)
            message_match = re.search(r'"message"\s*:\s*"([^"]+)"', content)

            args = {}
            if query_match:
                args["query"] = query_match.group(1)
            if user_id_match:
                args["user_id"] = user_id_match.group(1)
            if message_match:
                args["message"] = message_match.group(1)

            if args:
                return tool_name, args

    return None, None


def format_tool_call(
    tool_name: str, args: dict[str, Any], default_user_id: str = "4165"
) -> dict[str, Any] | None:
    """
    Format the output based on tool name with standardized OpenAI structure.

    Args:
        tool_name: Name of the tool
        args: Arguments dictionary
        default_user_id: Default user ID to use if not provided

    Returns:
        Formatted tool call dictionary or None if tool_name not recognized
    """
    if tool_name == "HandleOtherTalk":
        message = args.get("message", args.get("Message", args.get("query", "")))
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "HandleOtherTalk", "arguments": {"Message": message}}}
            ],
        }

    elif tool_name == "ToProductQAAssistant":
        query = args.get("query", args.get("properties", ""))
        if isinstance(query, dict):
            query = query.get("query", str(query))
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "ToProductQAAssistant", "arguments": {"query": query}}}
            ],
        }

    elif tool_name == "ToOrderStatusAssistant":
        query = args.get("query", "")
        user_id = args.get("user_id", default_user_id)
        if isinstance(query, dict):
            query = query.get("query", str(query))
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "ToOrderStatusAssistant",
                        "arguments": {"query": query, "user_id": str(user_id)},
                    },
                }
            ],
        }

    elif tool_name == "ToReturnProcessing":
        query = args.get("query", "")
        user_id = args.get("user_id", default_user_id)
        if isinstance(query, dict):
            query = query.get("query", str(query))
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "ToReturnProcessing",
                        "arguments": {"query": query, "user_id": str(user_id)},
                    },
                }
            ],
        }

    return None


def normalize_inference_results(
    input_path: Path | str, output_path: Path | str
) -> list[dict[str, Any]]:
    """
    Process and normalize inference results to standard OpenAI tool_calls format.

    Args:
        input_path: Path to input JSONL file with raw inference results
        output_path: Path to output JSONL file for normalized results

    Returns:
        List of normalized results
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    results: list[dict[str, Any]] = []
    skipped = 0

    with open(input_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                tool_name, args = extract_tool_info(record)

                if tool_name and args is not None:
                    formatted = format_tool_call(tool_name, args)
                    if formatted:
                        results.append(formatted)
                    else:
                        skipped += 1
                        logger.debug(f"Line {line_num}: Unrecognized tool '{tool_name}'")
                else:
                    skipped += 1
            except json.JSONDecodeError as e:
                skipped += 1
                logger.debug(f"Line {line_num}: JSON parse error - {e}")

    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    logger.info(f"Processed {len(results)} records, skipped {skipped}")
    return results

