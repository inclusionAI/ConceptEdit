# -*- coding: utf-8 -*-
"""
Template for config.py.

Copy this file to `config.py` (in the repository root) and fill in real
credentials. `config.py` is gitignored and must never be committed.

The pipeline modules import this file as a top-level module
(`from config import ...`).
"""

# ============================================================
# VLM / LLM service used by instruction generation & evaluation
# ============================================================
# An OpenAI-compatible HTTP endpoint (e.g. vLLM, SGLang, etc.)
# Used by:
#   - pipeline/prompt_single.py        (instruction generation, VLM)
#   - pipeline/prompt_multi.py        (instruction generation, VLM)
#   - pipeline/eval_metric.py          (VQA-based evaluation, LLM/VLM)
#   - pipeline/multi_eval_metric.py    (VQA-based evaluation, LLM/VLM)

# Endpoint that serves the VLM used for single-concept instruction generation
VLM_SINGLE_BASE_URL = "https://your-vlm-endpoint.example.com/v1"
VLM_SINGLE_API_KEY  = "EMPTY"
VLM_SINGLE_MODEL    = "your-vlm-model-name"  # or any local model name / path

# Endpoint that serves the VLM used for multi-concept instruction generation
VLM_MULTI_BASE_URL = "https://your-vlm-endpoint.example.com/v1"
VLM_MULTI_API_KEY  = "EMPTY"
VLM_MULTI_MODEL    = "your-vlm-model-name"

# Endpoint that serves the model used for VQA-based evaluation
EVAL_BASE_URL = "https://your-vlm-endpoint.example.com/v1"
EVAL_API_KEY  = "EMPTY"
EVAL_MODEL    = "your-vlm-model-name"

# ============================================================
# Editing model
# ============================================================
# Local checkpoint path (or HF repo id) loadable by diffusers.
# Used by pipeline/flux_edit.py and pipeline/multi_flux_edit.py.
FLUX_MODEL_PATH = "black-forest-labs/FLUX.2-klein"

# Number of GPUs to use for flux editing.
NUM_GPUS = 8

# ============================================================
# object-storage credentials (optional)
# ============================================================
# Only needed if your input images live on object storage. The pipeline also
# supports purely local image inputs, in which case these can stay
# as placeholders.
OBJECT_STORAGE_ACCESS_KEY_ID     = "your-object-storage-access-key-id"
OBJECT_STORAGE_ACCESS_KEY_SECRET = "your-object-storage-access-key-secret"
OBJECT_STORAGE_ENDPOINT          = "http://your-object-storage-endpoint.example.com"
