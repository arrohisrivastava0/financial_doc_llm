# LLM Fine-Tuning for Financial Sentiment Analysis

Fine-tuning SmolLM2-1.7B-Instruct on financial news data using LoRA/QLoRA for domain-adaptive sentiment classification.

## Overview
Large language models have strong general capabilities but underperform on domain-specific financial text. This project fine-tunes a 1.7B parameter LLM on financial news sentiment data using parameter-efficient fine-tuning (PEFT) with LoRA, making it practical to run on a single GPU.

## Approach
- **Model:** SmolLM2-1.7B-Instruct (HuggingFace)
- **Dataset:** Twitter Financial News Sentiment (9,543 training samples)
- **Fine-tuning method:** LoRA (r=16, alpha=32) via HuggingFace PEFT — only 0.18% of parameters trained
- **Quantization:** 4-bit NF4 via BitsAndBytes for memory-efficient training
- **Experiment tracking:** MLflow for logging hyperparameters and loss metrics
- **Task:** Instruction-tuned sentiment classification (bullish / bearish / neutral)

## Results
| Metric | Value |
|--------|-------|
| Trainable parameters | 3.1M / 1.71B (0.18%) |
| Training epochs | 2 |
| Final train loss | TBD |
| Eval loss | TBD |

## Setup
```bash
pip install transformers datasets peft trl accelerate bitsandbytes mlflow
```

## Usage
Open `finetune.ipynb` in Google Colab with a T4 GPU runtime.

## Why This Matters
Enterprise applications like SAP S/4HANA process large volumes of financial documents. Domain-adaptive LLM fine-tuning enables accurate extraction and classification of financial signals from unstructured text — a key capability for intelligent financial applications.
