# Image-Editing Concept Pipeline

A 3-stage pipeline for generating large-scale, taxonomy-grounded
image-editing datasets:

```
                ┌────────────────────┐
                │   1. Instruction   │  Sample concepts from a
  input images ─▶│       Generation   ├─▶ per-image JSON
                │   (VLM as author)  │   (+VQA test set)
                └────────────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   2. Image Edit    │  Run FLUX with the
                │      with FLUX     ├─▶  generated instruction
                └────────────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   3. VQA Evaluator │  Score each edit, decide
                │  (VLM as judge)    ├─▶  keep / discard / recaption
                └────────────────────┘
```

Two variants are shipped side-by-side:

| Variant | Per-image output | Use case |
|--------|-------------------|--------|
| **Single-concept** | one edit, one instruction | classic instruction-tuning data |
| **Multi-concept**  | 2–5 parallel edits bundled into one combined instruction | dense, multi-edit data |

---

## Repo layout

```
image_editing_pipeline/
├── config.example.py          # copy → config.py and fill in keys
├── data/
│   ├── taxonomy_single.json   # taxonomy used by single-concept generator
│   └── taxonomy_multi.json    # taxonomy used by multi-concept generator
├── pipeline/
│   ├── prompt_single.py       # VLM call: single-concept instruction author
│   ├── prompt_multi.py        # VLM call: multi-concept instruction author
│   ├── prompt_eval.py         # system/user prompts for the VQA judge
│   │
│   ├── instruct_gen.py        # step 1 — single-concept
│   ├── flux_edit.py           # step 2 — single-concept
│   ├── eval_metric.py         # step 3 — single-concept
│   │
│   ├── multi_instruct_gen.py  # step 1 — multi-concept
│   ├── multi_flux_edit.py     # step 2 — multi-concept
│   └── multi_eval_metric.py   # step 3 — multi-concept
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <this repo>
cd image_editing_pipeline

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# fill in model paths / API keys
cp config.example.py config.py
$EDITOR config.py
```

`config.py` is gitignored — never commit it.

You will need:

* an OpenAI-compatible VLM endpoint (e.g. [vLLM](https://github.com/vllm-project/vllm)
  or [SGLang](https://github.com/sgl-project/sglang) serving a vision-language model)
  for instruction generation **and** evaluation;
* a local [FLUX](https://huggingface.co/black-forest-labs) checkpoint
  loadable by 🤗 `diffusers`;
* (optional) object-storage credentials if your source images live in object storage;
  local file input is fully supported as well.

---

## Running the pipeline

All commands are run from the repo root (so that `config.py` is on the
Python path).

### Single-concept

```bash
# 1. Generate edit instructions
python -m pipeline.instruct_gen \
    --image-dir   /path/to/source_images \
    --taxonomy    data/taxonomy_single.json \
    --save-dir    /path/to/output

# 2. Run FLUX edits (pass one or more batch_<N>/ subfolders)
python -m pipeline.flux_edit /path/to/output/batch_0 /path/to/output/batch_1

# 3. VQA evaluation
python -m pipeline.eval_metric /path/to/output/batch_0 /path/to/output/batch_1
```

Output of each step lives next to its input:

```
batch_0/
├── 0_0_2.json                 # instruction + VQA test set
├── 0_0_2_edit.png             # FLUX edit result
└── 0_0_2_vqa_result.json      # judge verdict & recaption
```

### Multi-concept

Identical commands with the `multi_` prefix:

```bash
python -m pipeline.multi_instruct_gen \
    --image-dir /path/to/source_images \
    --taxonomy  data/taxonomy_multi.json \
    --save-dir  /path/to/output_multi

python -m pipeline.multi_flux_edit  /path/to/output_multi/batch_0
python -m pipeline.multi_eval_metric /path/to/output_multi/batch_0
```

`multi_instruct_gen.py` can also consume a JSONL of object-storage image paths via
`--jsonl` (one JSON object per line, with an `images` field). Use
`--help` for the full list of flags.

---

## Per-task JSON schema

### After step 1 (single)

```json
{
  "option_id": 2,
  "edit_concept": {"category": "...", "sub_category": "...", "task": "...", "detail": "..."},
  "instruction_en":  "...",
  "instruction_zh":  "...",
  "detailed_instruction_en": "...",
  "detailed_instruction_zh": "...",
  "is_chinese_text_edit": false,
  "evaluation_vqa": [ /* 5 binary questions */ ],
  "local_image_path": "..."
}
```

### After step 1 (multi)

```json
{
  "selected_option_ids": [0, 2, 90],
  "edit_concepts_used":  [ {...}, {...}, {...} ],
  "instruction_en":  "...",
  "detailed_instruction_en": "...",
  "evaluation_vqa": [ /* N + 4 binary questions */ ],
  ...
}
```

### After step 3 (both)

```json
{
  "source_json": "0_0_2.json",
  "overall_vqa_score": 0.8,
  "final_decision": {
    "keep": true,
    "recaption_prompt_en": "...",   // only filled if the original instruction missed the actual change
    "recaption_prompt_zh": "...",
    "reason": "..."
  },
  "vqa_details": [ /* per-question judgment */ ]
}
```

---

## Resume / fault tolerance

Every step is **idempotent and resume-safe**:

* `instruct_gen` skips images for which a JSON with the right prefix
  already exists;
* `flux_edit` skips JSONs whose `_edit.png` already exists;
* `eval_metric` skips JSONs whose `_vqa_result.json` already exists.

Killing the process and re-running picks up exactly where it left off.

---

## License

Released under the MIT License. See `LICENSE`.
