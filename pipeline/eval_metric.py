"""
Step 3 (single-concept): score each edited image with a VLM as judge.

For every ``<task_id>.json`` that has a corresponding
``<task_id>_edit.png``, this script asks an evaluator model to:

* answer the 5 binary VQA questions stored in the instruction JSON;
* decide whether the edit is worth keeping; and
* (optionally) produce a recaption that better describes the actual
  visual change.

The result is written to ``<task_id>_vqa_result.json`` next to the
input.

Concurrency model: two thread pools running as a pipeline.

  * PREP_WORKERS:   I/O + image resize/encode (CPU-bound bursts)
  * API_WORKERS:    network-bound model calls

Adjust the pool sizes via CLI flags.
"""

import argparse
import base64
import glob
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from urllib.parse import unquote, urlparse

from PIL import Image
from openai import OpenAI
from tqdm import tqdm

from pipeline.prompt_eval import get_eval_prompts
from config import (
    EVAL_BASE_URL,
    EVAL_API_KEY,
    EVAL_MODEL,
)


# ---- Optional object storage support --------------------------------------------------

def _try_import_object_storage():
    """Return an optional object-storage client.

    This sanitized release does not vendor any provider-specific storage SDK.
    Projects that need remote image loading can adapt this helper to their
    own storage backend and credentials outside the repository.
    """
    return None, None, None, None


_thread_local = threading.local()


def get_bucket(storage_mod, auth, endpoint, name):
    """Per-thread Bucket cache to avoid lock contention."""
    if not hasattr(_thread_local, "buckets"):
        _thread_local.buckets = {}
    if name not in _thread_local.buckets:
        _thread_local.buckets[name] = storage_mod.Bucket(auth, endpoint, name)
    return _thread_local.buckets[name]


def parse_storage_path(storage_path):
    parsed = urlparse(storage_path)
    if parsed.scheme == "storage":
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in ("http", "https"):
        return parsed.netloc.split(".")[0], unquote(parsed.path).lstrip("/")
    return None, None


def fetch_image_from_storage(bucket, object_key, max_retries=3):
    for attempt in range(max_retries):
        try:
            img_bytes = bucket.get_object(object_key).read()
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(0.5)


# ---- Image encoding -------------------------------------------------------

def encode_image_to_base64(image_obj, target_area=1024 * 1024):
    try:
        w, h = image_obj.size
        current_area = w * h
        if current_area > target_area:
            scale = (target_area / current_area) ** 0.5
            new_w, new_h = int(w * scale), int(h * scale)
            image_obj = image_obj.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buf = BytesIO()
        image_obj.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


# ---- Pipeline stages ------------------------------------------------------

def sync_prepare_data(task, storage_ctx):
    """Stage 1: load images, build the prompt payload."""
    json_path = task["json_path"]
    edit_img_path = task["edit_img_path"]
    save_path = task["save_path"]
    raw_img_path = task["image_path"]
    source_type = task["source_type"]

    if os.path.exists(save_path):
        return {"status": "skipped", "file": os.path.basename(json_path)}

    try:
        # 1. Load original image
        orig_img = None
        if source_type == "object_storage":
            storage_mod, storage_auth, storage_endpoint = storage_ctx
            if storage_mod is None:
                return {"status": "error",
                        "msg": "object storage source requested but object_storage_client unavailable",
                        "file": os.path.basename(json_path)}
            bucket_name, object_key = parse_storage_path(raw_img_path)
            bucket = get_bucket(storage_mod, storage_auth, storage_endpoint, bucket_name)
            orig_img = fetch_image_from_storage(bucket, object_key)
        elif source_type == "local":
            if os.path.exists(raw_img_path):
                orig_img = Image.open(raw_img_path).convert("RGB")

        if orig_img is None:
            return {"status": "error",
                    "msg": f"failed to load {source_type} image",
                    "file": os.path.basename(json_path)}

        # 2. Load edited image
        try:
            edit_img = Image.open(edit_img_path).convert("RGB")
        except Exception:
            return {"status": "error",
                    "msg": "failed to open edited image",
                    "file": os.path.basename(json_path)}

        # 3. Read metadata
        with open(json_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        instruction_en = meta_data.get("detailed_instruction_en", "")
        instruction_zh = meta_data.get("detailed_instruction_zh", "")
        vqa_list = meta_data.get("evaluation_vqa", [])
        edit_concept = meta_data.get("edit_concept", {})

        # 4. Match sizes, encode
        if orig_img.size != edit_img.size:
            orig_img = orig_img.resize(edit_img.size, Image.Resampling.LANCZOS)

        orig_b64 = encode_image_to_base64(orig_img)
        edit_b64 = encode_image_to_base64(edit_img)
        if not orig_b64 or not edit_b64:
            return {"status": "error",
                    "msg": "image encoding failed",
                    "file": os.path.basename(json_path)}

        concept_text = (
            f"Category: {edit_concept.get('category', 'N/A')} | "
            f"Sub-category: {edit_concept.get('sub_category', 'N/A')} | "
            f"Task: {edit_concept.get('task', 'N/A')} | "
            f"Detail: {edit_concept.get('detail', 'N/A')}"
        )
        questions_text = "".join(
            f"Question {i+1} (Dimension: {item.get('dimension', '')}): "
            f"{item.get('question_en', '')}\n"
            for i, item in enumerate(vqa_list)
        )

        system_prompt, user_content = get_eval_prompts(
            concept_text, instruction_en, instruction_zh,
            questions_text, orig_b64, edit_b64,
        )

        return {
            "status": "ready",
            "json_path": json_path,
            "save_path": save_path,
            "system_prompt": system_prompt,
            "user_content": user_content,
            "image_path": raw_img_path,
            "edit_concept": edit_concept,
            "instruction_en": instruction_en,
            "instruction_zh": instruction_zh,
            "vqa_list": vqa_list,
        }

    except Exception as e:
        return {"status": "error", "msg": str(e),
                "file": os.path.basename(json_path)}


def sync_request_api(prep_data, client, model_name):
    """Stage 2: call the evaluator, parse, save final JSON."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prep_data["system_prompt"]},
                {"role": "user", "content": prep_data["user_content"]},
            ],
            max_tokens=8192,
            temperature=0.2,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content
        gpt_response = json.loads(result_text)

        model_answers_map = {
            item["question_index"]: item for item in gpt_response.get("answers", [])
        }
        final_results = []
        overall_match_count = 0
        vqa_list = prep_data["vqa_list"]

        for i, vqa_item in enumerate(vqa_list):
            idx = i + 1
            question_text = vqa_item.get("question_en")
            expected_raw = str(vqa_item.get("expected_answer", "yes")).lower().strip().rstrip(".")

            model_res = model_answers_map.get(idx, {})
            model_judgment = str(model_res.get("quality_judgment", "unknown")).lower().strip()

            match = (model_judgment == "positive")
            if match:
                overall_match_count += 1

            final_results.append({
                "question_index": idx,
                "dimension": vqa_item.get("dimension"),
                "question": question_text,
                "expected_vqa_raw": expected_raw,
                "model_judgment": model_judgment,
                "match": match,
                "reason": model_res.get("reason", "No reason provided"),
            })

        score = overall_match_count / len(vqa_list) if vqa_list else 0
        model_final_decision = gpt_response.get("final_result", {})

        output_data = {
            "source_json": os.path.basename(prep_data["json_path"]),
            "image_path": prep_data["image_path"],
            "edit_concept": prep_data["edit_concept"],
            "instruction_en": prep_data["instruction_en"],
            "instruction_zh": prep_data["instruction_zh"],
            "overall_vqa_score": score,
            "final_decision": {
                "keep": model_final_decision.get("keep", False),
                "recaption_prompt_en": model_final_decision.get("recaption_prompt_en", ""),
                "recaption_prompt_zh": model_final_decision.get("recaption_prompt_zh", ""),
                "reason": model_final_decision.get("reason", ""),
            },
            "vqa_details": final_results,
        }

        with open(prep_data["save_path"], "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return {"status": "success", "file": os.path.basename(prep_data["save_path"])}

    except Exception as e:
        return {"status": "error",
                "msg": f"API error: {e}",
                "file": os.path.basename(prep_data["json_path"])}


# ---- Task discovery -------------------------------------------------------

def discover_tasks(target_dirs):
    tasks = []
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            continue
        for json_path in glob.glob(os.path.join(target_dir, "*.json")):
            if "_vqa_result.json" in json_path:
                continue
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            dir_path = os.path.dirname(json_path)
            edit_img_path = os.path.join(dir_path, f"{base_name}_edit.png")
            save_path = os.path.join(dir_path, f"{base_name}_vqa_result.json")

            if not (os.path.exists(edit_img_path) and not os.path.exists(save_path)):
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                storage_p = data.get("storage_image_path")
                local_p = data.get("local_image_path")
                image_path = source_type = None
                if storage_p:
                    image_path, source_type = storage_p, "object_storage"
                elif local_p:
                    image_path, source_type = local_p, "local"

                if image_path:
                    tasks.append({
                        "json_path": json_path,
                        "edit_img_path": edit_img_path,
                        "save_path": save_path,
                        "image_path": image_path,
                        "source_type": source_type,
                    })
            except Exception:
                continue
    return tasks


# ---- Main -----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VQA-based evaluation (single-concept)")
    p.add_argument("input_dirs", nargs="+", help="Directories produced by flux_edit.py")
    p.add_argument("--prep-workers", type=int, default=64,
                   help="Threads for image loading / resize / encode")
    p.add_argument("--api-workers", type=int, default=256,
                   help="Threads issuing concurrent API calls")
    p.add_argument("--base-url", default=EVAL_BASE_URL)
    p.add_argument("--api-key", default=EVAL_API_KEY)
    p.add_argument("--model", default=EVAL_MODEL)
    return p.parse_args()


def main():
    args = parse_args()

    storage_mod, ak, sk, storage_endpoint = _try_import_object_storage()
    storage_auth = storage_mod.Auth(ak, sk) if storage_mod is not None else None
    storage_ctx = (storage_mod, storage_auth, storage_endpoint)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    print("🔍 Scanning for evaluation tasks...")
    all_tasks = discover_tasks(args.input_dirs)
    print(f"📊 Found {len(all_tasks)} pending tasks.")
    if not all_tasks:
        return

    success_cnt = fail_cnt = 0
    stats_lock = threading.Lock()

    prep_executor = ThreadPoolExecutor(max_workers=args.prep_workers)
    api_executor = ThreadPoolExecutor(max_workers=args.api_workers)

    print("🚀 Starting evaluation pipeline...")
    prep_futures = {prep_executor.submit(sync_prepare_data, t, storage_ctx): t for t in all_tasks}

    with tqdm(total=len(all_tasks), desc="Evaluating", unit="pair") as pbar:

        def handle_api_result(future):
            nonlocal success_cnt, fail_cnt
            res = future.result()
            with stats_lock:
                if res["status"] == "success":
                    success_cnt += 1
                else:
                    fail_cnt += 1
                    tqdm.write(f"❌ API error [{res.get('file')}]: {res.get('msg')}")
                pbar.update(1)
                pbar.set_postfix({"OK": success_cnt, "Fail": fail_cnt})

        for fut in as_completed(prep_futures):
            prep_res = fut.result()
            if prep_res["status"] == "skipped":
                with stats_lock:
                    pbar.update(1)
                continue
            if prep_res["status"] == "error":
                with stats_lock:
                    fail_cnt += 1
                    tqdm.write(f"❌ prep error [{prep_res.get('file')}]: {prep_res.get('msg')}")
                    pbar.update(1)
                continue
            api_fut = api_executor.submit(sync_request_api, prep_res, client, args.model)
            api_fut.add_done_callback(handle_api_result)

    api_executor.shutdown(wait=True)
    prep_executor.shutdown(wait=True)

    print(f"\n✅ Done. success={success_cnt}, failed={fail_cnt}")


if __name__ == "__main__":
    main()
