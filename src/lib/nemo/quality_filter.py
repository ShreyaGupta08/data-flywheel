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
Quality filtering utilities for evaluating and filtering inference results
using LLM-as-judge evaluation.
"""

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.quality_filter")


# System prompt for tool-calling quality evaluation
DATA_QUALITY_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for tool-calling responses.
Your task is to evaluate if the model called the CORRECT tool with appropriate arguments.

EVALUATION CRITERIA:
1. CORRECT TOOL SELECTION:
   - Order status, delivery, shipping inquiries → ToOrderStatusAssistant
   - Return, refund, return processing → ToReturnProcessing  
   - Product info, specs, features, inventory → ToProductQAAssistant
   - Off-topic, greetings, unrelated questions → HandleOtherTalk

2. ARGUMENT QUALITY:
   - query: Should capture the user's intent accurately
   - user_id: Should be present when required (for order/return tools)

Remember, different tools need different arguments.
- HandleOtherTalk needs a message argument
- ToProductQAAssistant needs a query argument
- ToOrderStatusAssistant needs a query and user_id arguments
- ToReturnProcessing needs a query and user_id arguments

Return ONLY:
- 1 if the tool call is correct (right function AND reasonable arguments)
- 0 if the tool call is incorrect (wrong function OR missing/invalid arguments)"""

DATA_QUALITY_JUDGE_USER_TEMPLATE = """System Context: {{item.system_context}}

User Query: {{item.user_query}}

Available Tools: {{item.available_tools}}

Model's Tool Call Response: {{item.model_response}}

Is this tool call correct? (1 = correct, 0 = incorrect):"""


def get_data_quality_eval_config(judge_model: dict[str, Any]) -> dict[str, Any]:
    """
    Generate evaluation configuration for data quality LLM-as-judge evaluation.

    This config is used to evaluate tool-calling correctness on inference results
    that have been pre-processed into evaluation format.

    Args:
        judge_model: Dictionary with API endpoint configuration, e.g.:
            {
                "api_endpoint": {
                    "url": "https://integrate.api.nvidia.com/v1/chat/completions",
                    "model_id": "meta/llama-3.1-70b-instruct",
                    "api_key": "your-api-key"
                }
            }

    Returns:
        Evaluation configuration dictionary for NeMo Evaluator
    """
    return {
        "type": "custom",
        "tasks": {
            "data-quality-check": {
                "type": "data",
                "metrics": {
                    "tool-correctness": {
                        "type": "llm-judge",
                        "params": {
                            "model": judge_model,
                            "template": {
                                "messages": [
                                    {"role": "system", "content": DATA_QUALITY_JUDGE_SYSTEM_PROMPT},
                                    {"role": "user", "content": DATA_QUALITY_JUDGE_USER_TEMPLATE},
                                ]
                            },
                            "scores": {
                                "is_correct": {
                                    "type": "integer",
                                    "parser": {"type": "regex", "pattern": "([01])"},
                                }
                            },
                        },
                    }
                },
            }
        },
    }


def extract_high_quality_samples(
    evaluator_url: str,
    job_id: str,
    eval_data_path: Path | str,
    output_path: Path | str,
    client_id: str = "filtered",
    model_name: str = "meta/llama-3.2-1b-instruct",
) -> list[dict[str, Any]]:
    """
    Extract samples that passed LLM judge evaluation (score=1).

    Parses evaluation job logs to identify which samples received a score of 1,
    then extracts and formats those samples for use in training.

    Args:
        evaluator_url: Base URL of the NeMo evaluator service
        job_id: Completed evaluation job ID
        eval_data_path: Path to original evaluation dataset (JSONL)
        output_path: Path to save high-quality samples (JSONL)
        client_id: Client ID for output records
        model_name: Model name to use in output records

    Returns:
        List of high-quality sample records in training format
    """
    eval_data_path = Path(eval_data_path)
    output_path = Path(output_path)

    # Get logs to extract per-sample scores
    logs_response = requests.get(
        f"{evaluator_url}/jobs/{job_id}/logs", headers={"accept": "application/json"}
    )

    if logs_response.status_code != 200:
        raise RuntimeError(f"Failed to get evaluation logs: {logs_response.status_code}")

    log_text = logs_response.text
    score_pattern = r"Computed metric (\d+) tool-correctness:.*is_correct.*value=(\d+\.?\d*)"
    matches = re.findall(score_pattern, log_text)

    high_quality_indices: list[int] = []
    for sample_idx, score in matches:
        if float(score) == 1.0:
            high_quality_indices.append(int(sample_idx))

    logger.info(f"Found {len(high_quality_indices)} high-quality samples from {len(matches)} evaluated")

    # Load original eval data
    eval_data: list[dict[str, Any]] = []
    with open(eval_data_path, "r") as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))

    # Extract and format high-quality samples
    high_quality_samples: list[dict[str, Any]] = []
    for idx in high_quality_indices:
        if idx < len(eval_data):
            sample = eval_data[idx]
            inference_result = {
                "request": {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": sample["system_context"]},
                        {"role": "user", "content": sample["user_query"]},
                    ],
                    "tools": sample["available_tools"],
                },
                "response": {
                    "choices": [
                        {"message": {"role": "assistant", "content": sample["model_response"]}}
                    ]
                },
                "workload_id": "primary_assistant",
                "client_id": client_id,
                "timestamp": int(time.time()),
            }
            high_quality_samples.append(inference_result)

    # Save to file
    with open(output_path, "w") as f:
        for sample in high_quality_samples:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Saved {len(high_quality_samples)} high-quality samples to {output_path}")
    return high_quality_samples


def wait_for_eval_job_simple(
    evaluator_url: str, job_id: str, polling_interval: int = 10, timeout: int = 3600
) -> dict[str, Any]:
    """
    Wait for evaluation job to complete (simplified version for notebooks).

    Args:
        evaluator_url: Base URL of the evaluator service
        job_id: Job ID to monitor
        polling_interval: Seconds between status checks
        timeout: Maximum seconds to wait

    Returns:
        Job data dictionary

    Raises:
        TimeoutError: If job doesn't complete within timeout
        RuntimeError: If job fails
    """
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

        response = requests.get(f"{evaluator_url}/jobs/{job_id}")
        job_data = response.json()
        status = job_data.get("status", "unknown")

        if status == "completed":
            logger.info("✓ Job completed!")
            return job_data
        elif status == "failed":
            raise RuntimeError(f"Job failed: {job_data}")
        elif status == "running":
            progress = job_data.get("status_details", {}).get("progress", 0)
            logger.info(f"Running... {progress:.1f}%")
        else:
            logger.info(f"Status: {status}")

        time.sleep(polling_interval)

