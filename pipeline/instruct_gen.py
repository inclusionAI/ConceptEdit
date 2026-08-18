"""
Step 1 (single-concept): generate per-image editing instructions.

Scans a directory of local images, samples a diverse set of candidate
concepts from the taxonomy for each image, and uses a multimodal LLM
(via ``prompt_single.generate_multi_edit_data``) to produce structured
JSON edit instructions plus a VQA test set.

Each output JSON is stored under ``<save_dir>/batch_<N>/`` so a single
directory never grows too large.

Run::

    python -m pipeline.instruct_gen \
        --image-dir   /path/to/images \
        --taxonomy    data/taxonomy_single.json \
        --save-dir    /path/to/save

(or simply ``python pipeline/instruct_gen.py ...`` when the working
directory contains ``config.py``).
"""

import argparse
import base64
import concurrent.futures
import io
import json
import os
import random
import time

from PIL import Image
from tqdm import tqdm

from pipeline.prompt_single import generate_multi_edit_data


# ---- Defaults -------------------------------------------------------------

DEFAULT_MAX_WORKERS = 64
DEFAULT_CANDIDATE_COUNT = 10
DEFAULT_MAX_RETRIES = 2
DEFAULT_BATCH_SIZE = 10000   # JSONs per batch_<N>/ sub-directory


# ---- Taxonomy helpers ------------------------------------------------------

def load_concept_data_hierarchical(json_file_path):
    """Load the taxonomy JSON and flatten it into ``{top_id: [concept,...]}``."""
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        categorized_data = {}
        for category in raw_data["categories"]:
            top_id = category["id"]
            top_name = category["name"]
            categorized_data.setdefault(top_id, [])

            for child in category.get("children", []):
                sub_name = child["name"]
                for task in child.get("tasks", []):
                    task_name = task["name"]
                    details = task.get("details_zh", [])
                    if not details:
                        continue
                    for detail in details:
                        categorized_data[top_id].append({
                            "top_id": top_id,
                            "category_name": top_name,
                            "sub_category": sub_name,
                            "task": task_name,
                            "detail": detail,
                        })
        return categorized_data
    except Exception as e:
        print(f"Failed to parse taxonomy JSON: {e}")
        return None


def extract_diverse_concepts(categorized_data, count=6):
    """Sample ``count`` concepts, biased toward diversity across top-level pillars."""
    concepts = []
    pillar_ids = list(categorized_data.keys())
    random.shuffle(pillar_ids)
    for pid in pillar_ids:
        if len(concepts) >= count:
            break
        if categorized_data[pid]:
            concepts.append(random.choice(categorized_data[pid]))

    all_items = [c for pid in categorized_data for c in categorized_data[pid]]
    while len(concepts) < count and all_items:
        c = random.choice(all_items)
        if c not in concepts:
            concepts.append(c)

    random.shuffle(concepts)
    return concepts


# ---- Image encoding --------------------------------------------------------

def encode_image_from_path(image_path, target_area=1024 * 1024):
    """Load an image, normalize to ~1MP area, return base64 PNG."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            current_area = w * h
            if current_area != target_area:
                scale = (target_area / current_area) ** 0.5
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return None


# ---- Per-image worker ------------------------------------------------------

def process_single_image_local(task_idx, image_path, filename_no_ext,
                               categorized_db, save_dir,
                               candidate_count, max_retries, batch_size):
    task_id = f"Idx {task_idx} - {filename_no_ext}"
    try:
        img_b64 = encode_image_from_path(image_path)
        if not img_b64:
            return False, 0, f"[{task_id}] image encoding failed"

        last_error = "unknown error"

        for _attempt in range(max_retries):
            concepts = extract_diverse_concepts(categorized_db, count=candidate_count)
            if not concepts:
                return False, 0, f"[{task_id}] empty concept sample"

            result_list = generate_multi_edit_data(concepts, img_b64)

            if isinstance(result_list, list):
                if len(result_list) > 0:
                    saved_count = 0
                    batch_folder_idx = task_idx // batch_size
                    batch_save_dir = os.path.join(save_dir, f"batch_{batch_folder_idx}")
                    os.makedirs(batch_save_dir, exist_ok=True)

                    for item in result_list:
                        if "edit_concept" not in item:
                            continue

                        item["local_image_path"] = image_path
                        item["source_info"] = {"task_idx": task_idx, "filename": filename_no_ext}

                        raw_option_id = item.get("option_id", -1)
                        is_valid_candidate = (
                            isinstance(raw_option_id, int)
                            and 0 <= raw_option_id < candidate_count
                        )
                        is_fallback = (raw_option_id == 99)

                        if is_valid_candidate:
                            suffix = str(raw_option_id)
                        elif is_fallback:
                            suffix = "auto"
                        else:
                            continue

                        json_filename = f"{task_idx}_0_{suffix}.json"
                        json_save_path = os.path.join(batch_save_dir, json_filename)

                        try:
                            with open(json_save_path, "w", encoding="utf-8") as f:
                                json.dump(item, f, indent=2, ensure_ascii=False)
                            saved_count += 1
                        except Exception as e:
                            last_error = f"json save failed: {e}"

                    if saved_count > 0:
                        return True, saved_count, f"[{task_id}] -> {saved_count} JSONs"
                    last_error = "API returned a list but no valid items"
                    continue
                last_error = "API returned an empty list"
                continue

            if isinstance(result_list, dict) and "error" in result_list:
                error_msg = result_list["error"]
                last_error = f"API error: {error_msg}"
                if "Connection error" in error_msg or "429" in error_msg:
                    tqdm.write(f"⚠️ [{task_id}] API busy, sleeping 30s...")
                    time.sleep(30)
                else:
                    time.sleep(1)
                continue

            last_error = "Unexpected API response format"

        return False, 0, f"[{task_id}] retries exhausted: {last_error}"

    except Exception as e:
        return False, 0, f"[{task_id}] worker exception: {e}"


# ---- Main -----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate single-concept edit instructions")
    p.add_argument("--image-dir", required=True, help="Directory of input .jpg/.png images")
    p.add_argument("--taxonomy", required=True, help="Path to taxonomy JSON")
    p.add_argument("--save-dir", required=True, help="Where to write generated JSONs")
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--no-resume", action="store_true",
                   help="Disable breakpoint-resume scan (default: enabled)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading taxonomy...")
    concept_db = load_concept_data_hierarchical(args.taxonomy)
    if not concept_db:
        return

    print(f"Scanning images in {args.image_dir} ...")
    all_files = sorted([
        f for f in os.listdir(args.image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    all_tasks = []
    for idx, filename in enumerate(all_files):
        all_tasks.append({
            "task_idx": idx,
            "image_path": os.path.join(args.image_dir, filename),
            "filename_no_ext": os.path.splitext(filename)[0],
        })

    total_images = len(all_tasks)
    print(f"Found {total_images} images.")

    if not args.no_resume:
        print("Scanning for completed tasks (resume mode)...")
        pending = []
        dir_cache = {}
        for task in tqdm(all_tasks, desc="resume scan"):
            t_idx = task["task_idx"]
            batch_idx = t_idx // args.batch_size
            batch_dir = os.path.join(args.save_dir, f"batch_{batch_idx}")
            if batch_idx not in dir_cache:
                dir_cache[batch_idx] = set(os.listdir(batch_dir)) if os.path.exists(batch_dir) else set()
            prefix = f"{t_idx}_0_"
            if not any(f.startswith(prefix) for f in dir_cache[batch_idx]):
                pending.append(task)
        print(f"Resume: skipping {total_images - len(pending)}, remaining {len(pending)}.")
        all_tasks = pending

    if not all_tasks:
        print("🎉 Nothing left to do.")
        return

    print(f"\n--- Launching pool (workers={args.max_workers}) ---")

    total_jsons = success_img = fail_img = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(
                process_single_image_local,
                t["task_idx"], t["image_path"], t["filename_no_ext"],
                concept_db, args.save_dir,
                args.candidate_count, args.max_retries, args.batch_size,
            ): t for t in all_tasks
        }

        pbar = tqdm(concurrent.futures.as_completed(future_to_task),
                    total=len(all_tasks), unit="img")
        for future in pbar:
            try:
                success, count, msg = future.result()
                if success:
                    success_img += 1
                    total_jsons += count
                else:
                    fail_img += 1
                    tqdm.write(f"❌ {msg}")
            except Exception as e:
                fail_img += 1
                tqdm.write(f"❌ worker crashed: {e}")

            pbar.set_postfix({"OK": success_img, "Fail": fail_img, "JSONs": total_jsons})

    print(f"\nDone. success={success_img}, failed={fail_img}, jsons={total_jsons}.")


if __name__ == "__main__":
    main()
