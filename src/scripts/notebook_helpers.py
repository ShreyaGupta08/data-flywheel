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
Helper functions specifically for Jupyter notebook tutorials.
These provide simplified interfaces to the core flywheel functionality.
"""

import json
import time
from pathlib import Path
from typing import Any


def prepare_eval_dataset(
    inference_results_path: Path | str, output_path: Path | str
) -> list[dict[str, Any]]:
    """
    Transform inference results to format for LLM-as-judge evaluation.

    Extracts system context, user query, available tools, and model response
    from inference results and saves them in a format suitable for the
    NeMo Evaluator LLM-as-judge evaluation.

    Args:
        inference_results_path: Path to inference results JSONL file
        output_path: Path to save evaluation dataset JSONL

    Returns:
        List of evaluation records
    """
    inference_results_path = Path(inference_results_path)
    output_path = Path(output_path)

    eval_dataset: list[dict[str, Any]] = []

    with open(inference_results_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                messages = record["request"]["messages"]

                system_messages = [msg for msg in messages if msg["role"] == "system"]
                system_context = system_messages[0]["content"] if system_messages else ""

                user_messages = [msg for msg in messages if msg["role"] == "user"]
                user_query = user_messages[-1]["content"] if user_messages else ""

                tools = record["request"].get("tools", [])
                response_msg = record["response"]["choices"][0]["message"]
                model_response = json.dumps(
                    response_msg.get("tool_calls", response_msg.get("content", ""))
                )

                eval_record = {
                    "system_context": system_context,
                    "user_query": user_query,
                    "available_tools": tools,
                    "model_response": model_response,
                    "original_record": record,
                }
                eval_dataset.append(eval_record)

    with open(output_path, "w") as f:
        for record in eval_dataset:
            f.write(json.dumps(record) + "\n")

    print(f"Prepared {len(eval_dataset)} records for evaluation")
    return eval_dataset


def convert_to_training_format(
    input_data_list: list[dict[str, Any]],
    results: list[dict[str, Any]],
    output_path: Path | str,
    model_name: str = "meta/llama-3.2-1b-instruct",
) -> list[dict[str, Any]]:
    """
    Convert normalized inference results to training data format.

    Combines original input data with normalized results to create
    records in the standard flywheel training format.

    Args:
        input_data_list: Original input data list
        results: Normalized inference results
        output_path: Path to save formatted training data
        model_name: Model name to use in records

    Returns:
        List of formatted training records
    """
    output_path = Path(output_path)

    formatted_data: list[dict[str, Any]] = []
    for i, input_data in enumerate(input_data_list):
        if i >= len(results):
            break
        result = results[i]

        record = {
            "request": {
                "model": model_name,
                "messages": input_data["request"]["messages"],
                "tools": input_data["request"]["tools"],
            },
            "response": {"choices": [{"message": result}]},
            "workload_id": input_data.get("workload_id", "primary_assistant"),
            "client_id": input_data.get("client_id", "inference-test"),
            "timestamp": int(time.time()),
        }
        formatted_data.append(record)

    with open(output_path, "w") as f:
        for record in formatted_data:
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(formatted_data)} records to {output_path}")
    return formatted_data


def check_tool_calls_format(file_path: Path | str, max_samples: int = 5) -> None:
    """
    Verify data samples have proper tool_calls structure.

    Useful for debugging to check if inference results have the expected format.

    Args:
        file_path: Path to JSONL file to check
        max_samples: Maximum samples to check
    """
    file_path = Path(file_path)

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            record = json.loads(line)
            msg = record["response"]["choices"][0]["message"]
            has_tool_calls = "tool_calls" in msg
            has_content = "content" in msg
            print(f"Sample {i+1}: tool_calls={has_tool_calls}, content={has_content}")


def load_jsonl(file_path: Path | str) -> list[dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON records
    """
    file_path = Path(file_path)
    records: list[dict[str, Any]] = []

    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_jsonl(records: list[dict[str, Any]], file_path: Path | str) -> None:
    """
    Save a list of dictionaries to a JSONL file.

    Args:
        records: List of records to save
        file_path: Path to output JSONL file
    """
    file_path = Path(file_path)

    with open(file_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(records)} records to {file_path}")

