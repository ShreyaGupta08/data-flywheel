# NVIDIA Data Flywheel Foundational Blueprint Notebooks

## Introduction

A data flywheel creates a self-reinforcing loop where user interactions continuously enhance the AI application. As users engage, their input helps identify more efficient models (or NIMs) that achieve comparable accuracy, reducing the total cost of ownership. Additionally, these interactions can help improve or maintain model accuracy, leading to better responses and contribute to the ongoing generation of higher-quality data.

![Data Flywheel](./img/dfw-diagram.png)

Key steps in a data flywheel include logging user interactions, processing the data, customizing and evaluating candidate models, adding guardrails, and integrating them with external knowledge bases for question answering.

## Use Case: AI Virtual Assistant for Customer Service

These tutorials use sample data from an [AI Virtual Customer Service Assistant](https://build.nvidia.com/nvidia/ai-virtual-assistant-for-customer-service) that employs tool calling to route user queries to specialized assistants:

- **Product Q&A** - Answer questions about product specifications and features
- **Order Status** - Check delivery and shipping status
- **Returns Processing** - Handle return and refund requests
- **Small Talk** - Manage casual conversation and off-topic queries

The goal is to fine-tune smaller, more cost-efficient models (e.g., `meta/llama-3.2-1B-instruct`) to match the accuracy of larger deployed models (e.g., `meta/llama-3.3-70B-instruct`) on these tool-calling tasks.

## How it Works

The Data Flywheel Blueprint provides a unified API (the Orchestrator) that abstracts away the complexity of directly managing [NVIDIA NeMo Microservices](https://docs.nvidia.com/nemo/microservices/latest/about/index.html). As a user, you interact only with the Data Flywheel Orchestrator API to:

- Launch new flywheel jobs (for fine-tuning, evaluation, and deployment of models)
- Monitor job progress and results
- Access evaluation metrics and customization status

**What happens under the hood:**  

When you submit a job via the Orchestrator API, the system:
- Retrieves and processes your data (e.g., from Elasticsearch)
- Creates and uploads datasets for training, evaluation, and validation
- Deploys and manages candidate models (NIMs) on the NeMo platform
- Runs evaluations (including LLM-as-a-judge if configured)
- Handles model customization and tracks progress
- Aggregates results and exposes them through the same API

All direct communication with the NeMo Microservices platform (model deployment, evaluation, customization, etc.) is handled by the orchestrator's backend services. This means you do not need to interact with NeMo APIs or infrastructure directly—the orchestrator manages the full workflow for you.

## Notebooks

Check out the following example notebooks to learn how to optimize LLMs using a data flywheel.

### Core Tutorials

- **[Discover More Cost-Efficient AI Customer Service Agents](./data-flywheel-bp-tutorial.ipynb)**: The quickstart tutorial that demonstrates the end-to-end Data Flywheel workflow using the Orchestrator API. Load sample data, create a flywheel job, and monitor the customization and evaluation process.

- **[Feedback Loop Tutorial](./data-flywheel-bp-feedback-loop-tutorial.ipynb)**: A detailed, step-by-step tutorial that walks through the complete feedback loop process for continuous model improvement:
  
  | Step | Description |
  |------|-------------|
  | **0. Setup** | Configure the Data Flywheel Blueprint and NeMo Microservices |
  | **1. Load Sample Data** | Load tool-calling logs from the AI Virtual Assistant |
  | **2. Run Base Model Inference** | Generate outputs using a smaller base model (e.g., Llama-3.2-1B) |
  | **3. Data Quality Filtering** | Apply quality checks including output normalization and LLM-as-Judge evaluation to extract high-quality training samples |
  | **4. Load into Elasticsearch** | Store filtered data for the Flywheel service |
  | **5. Run Flywheel Job** | Execute model customization with the curated data |

  This notebook is ideal for understanding how to build a data quality pipeline that filters and validates model outputs before using them for fine-tuning.

### Which Notebook Should I Use?

| If you want to... | Use this notebook |
|-------------------|-------------------|
| Get started quickly with the end-to-end workflow | [data-flywheel-bp-tutorial.ipynb](./data-flywheel-bp-tutorial.ipynb) |
| Understand data quality filtering and the feedback loop in detail | [data-flywheel-bp-feedback-loop-tutorial.ipynb](./data-flywheel-bp-feedback-loop-tutorial.ipynb) |
| Learn how to run inference and evaluate outputs before training | [data-flywheel-bp-feedback-loop-tutorial.ipynb](./data-flywheel-bp-feedback-loop-tutorial.ipynb) |

## Prerequisites

### Hardware Requirement

To complete this tutorial, you'll need a system with at least two A100 or H100 (80GB) NVIDIA GPUs, which will be used as follows:

- **Fine-tuning:** At least one GPU is required for fine-tuning a model (e.g.`meta/llama-3.2-1B-instruct`, `meta/llama-3.2-3B-instruct` or `meta/llama-3.1-8B-instruct`).
- **Inference:** At least one GPU is required for deploying the corresponding NIM for evaluation.

### Software Requirement

You will deploy the [NVIDIA NeMo Microservices](https://docs.nvidia.com/nemo/microservices/latest/about/index.html) as part of this blueprint.

First, please ensure your platform meets the [Requirements](https://docs.nvidia.com/nemo/microservices/latest/get-started/platform-prereq.html#requirements) before proceeding. The notebook uses a script to automate the remaining setup, including the minikube cluster and NeMo microservices deployment.


### Get the Data Flywheel Blueprint

1. Clone the blueprint repository:

   ```sh
   git clone git@github.com:NVIDIA-AI-Blueprints/data-flywheel.git

   cd data-flywheel
   ```

2. Install dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/):

   ```sh
   uv sync --dev
   ```

### Access the Tutorial

1. Launch Jupyter Lab to begin working with the provided tutorial.

   ```bash
   uv run --with jupyter jupyter lab --ip=0.0.0.0
   ```

2. Navigate to the [notebook](#notebooks).
