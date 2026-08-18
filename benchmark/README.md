# ConceptBench

ConceptBench is an image editing benchmark with 1,000 curated test cases across 6 major editing categories. Each case provides a source image, English/Chinese edit instructions, detailed edit instructions, and hierarchical task metadata.

## Repository Structure

```text
benchmark/
├── README.md                  # Usage, data format, and evaluation guide
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignores data/, results/, caches, and environment files
└── scripts/
    ├── infer.py               # Flux2Klein inference example
    ├── infer_qwen.py          # Qwen-Image-Edit inference example
    ├── eval.py                # OpenAI-compatible VLM judge evaluation
    ├── benchmark_prompts.py   # IF/VQ/NC evaluation prompts
    └── visualize.py           # Gradio visualization for evaluation reports
```

## Data Placement

The benchmark data is not included in this code repository. Please download the data separately and place the extracted `data/` directory under the benchmark root.

After placement, the directory layout should be:

```text
/path/to/benchmark/data/taxonomy.json
/path/to/benchmark/data/images/*.jpg
/path/to/benchmark/data/<category>/<sub_category>/<task>/<detail>.json
```

Expected data size:

- `1,000` case JSON files under `data/<category>/<sub_category>/<task>/<detail>.json`
- `979` source images under `data/images/`
- Each JSON uses a `local_image_path` relative to `data/`, for example `images/example.jpg`

The scripts read benchmark cases from `./data` and write model outputs and reports to `./results` by default.

## Benchmark JSON Format

Each case JSON keeps only the fields required for inference and evaluation:

```json
{
  "caption": "source image caption",
  "edit_concept": {
    "category": "...",
    "sub_category": "...",
    "task": "...",
    "detail": "..."
  },
  "instruction_en": "short English edit instruction",
  "instruction_zh": "short Chinese edit instruction",
  "detailed_instruction_en": "detailed English instruction used by inference/evaluation",
  "detailed_instruction_zh": "detailed Chinese instruction",
  "local_image_path": "images/example.jpg"
}
```

Field notes:

- `local_image_path` is resolved relative to `data/`.
- `detailed_instruction_en` is used by default for inference and VLM-judge evaluation; the scripts fall back to `instruction_en` if it is missing.
- Source sampling metadata, internal paths, object-storage paths, intermediate reasoning traces, historical VQA results, VQA questions, and text-edit flags are intentionally removed from the public JSON files.

## Installation

```bash
pip install -r requirements.txt
```

If you use the model pipelines in the example scripts, install a `diffusers` version that supports the corresponding pipelines:

```bash
cd /path/to/diffusers
pip install -e .
```

## Inference

### Flux2Klein Example

```bash
cd /path/to/benchmark
python scripts/infer.py \
  --model-name Flux2Klein-9b \
  --model-path /path/to/FLUX.2-klein-9B \
  --num-gpus 8
```

### Qwen-Image-Edit Example

```bash
cd /path/to/benchmark
python scripts/infer_qwen.py \
  --model-name Qwen-Image-Edit-2511 \
  --model-path /path/to/Qwen-Image-Edit-2511 \
  --num-gpus 8
```

Generated images are saved under `results/{model_name}/` with the same hierarchy as the case JSON files:

```text
results/Flux2Klein-9b/<category>/<sub_category>/<task>/<detail>.png
```

## Evaluation

`eval.py` uses an OpenAI-compatible Chat Completions API as a VLM judge. Configure credentials through environment variables or command-line arguments. Do not hard-code API keys in the code.

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional
export VLM_MODEL="gpt-4.1"                          # Optional

python scripts/eval.py \
  --model-name Flux2Klein-9b \
  --threads 20
```

You can also pass the API settings directly:

```bash
python scripts/eval.py \
  --model-name Flux2Klein-9b \
  --api-key "your_api_key" \
  --base-url "https://api.openai.com/v1" \
  --judge-model "gpt-4.1" \
  --threads 20
```

The evaluation report is written to:

```text
results/{model_name}/benchmark_report.json
```

## Metrics

Each edited image is scored from 0 to 10 on three dimensions:

| Metric | Description |
|---|---|
| IF (Instruction Following) | Whether the edit follows the requested instruction |
| VQ (Visual Quality) | Overall image quality, naturalness, and visual fidelity |
| NC (Non-edited Consistency) | Whether non-edited regions remain consistent with the source image |

The per-case score is the average of IF, VQ, and NC.

## Evaluation Report Format

`benchmark_report.json` contains the overall score, hierarchical category scores, and detailed per-case records:

```json
{
  "total_avg": 8.62,
  "categories": {
    "portrait_human_specialized": {
      "sub_categories": {}
    }
  },
  "detailed_samples": [
    {
      "info": {
        "cat": "portrait_human_specialized",
        "subcat": "face_editing",
        "task_name": "hairstyle_editing",
        "detail": "Bob_Cut",
        "prompt": "..."
      },
      "metrics": {
        "if": 8.5,
        "vq": 8.4,
        "nc": 8.9
      },
      "avg": 8.6,
      "full": {
        "if_reason": "...",
        "vq_reason": "...",
        "nc_reason": "..."
      }
    }
  ]
}
```

## Visualization

```bash
python scripts/visualize.py \
  --model-name Flux2Klein-9b \
  --port 7860
```

Open the local Gradio URL printed by the script to inspect the evaluation report interactively.

## Notes

- Make sure GPUs, model weights, and the required model pipelines are available before running inference.
- Evaluation requires access to the configured OpenAI-compatible API service.
- Inference and evaluation support resumable runs: existing generated images or report entries are skipped/reused according to the script logic.
- The released benchmark data contains source images and edit instructions only; generated model outputs should be produced by running the inference scripts.
