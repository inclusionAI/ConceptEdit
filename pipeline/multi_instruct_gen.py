"""
Step 1 (multi-concept): generate multi-edit instruction sets.

For each input image, the multi-concept generator samples a larger pool
of candidate concepts from the taxonomy and asks the VLM to assemble
2-5 parallel edits per instruction set, returning multiple distinct
sets per image.

Input formats supported:

* ``--image-dir <dir>``: a flat directory of local images
* ``--jsonl <path>``: a JSONL file where each line has an ``images``
  field (list of dicts with ``image_path``) pointing to object storage URIs
  (``storage://bucket/key``) or HTTP(S) URLs. Requires object-storage credentials in
  ``config.py``.

Each output JSON is stored under ``<save_dir>/batch_<N>/`` with a
filename ``<line_idx>_<img_idx>_<id1_id2_id3>.json``.
"""

import argparse
import base64
import concurrent.futures
import io
import json
import os
import random
import time
from urllib.parse import unquote, urlparse

from PIL import Image
from tqdm import tqdm

from pipeline.prompt_multi import generate_multi_edit_data


# ---- Defaults -------------------------------------------------------------

DEFAULT_MAX_WORKERS = 64
DEFAULT_CANDIDATE_COUNT = 16
DEFAULT_MAX_RETRIES = 2
DEFAULT_BATCH_SIZE = 10000


# ---- Optional object storage support --------------------------------------------------

def _try_import_object_storage():
    """Return an optional object-storage client.

    This sanitized release does not vendor any provider-specific storage SDK.
    Projects that need remote image loading can adapt this helper to their
    own storage backend and credentials outside the repository.
    """
    return None, None, None, None


def parse_storage_path(storage_path):
    parsed = urlparse(storage_path)
    if parsed.scheme == "storage":
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in ("http", "https"):
        return parsed.netloc.split(".")[0], unquote(parsed.path).lstrip("/")
    return None, None


# ---- Taxonomy helpers ------------------------------------------------------

def load_concept_data_hierarchical(json_file_path):
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        out = {}
        for category in raw_data["categories"]:
            top_id = category["id"]
            top_name = category["name"]
            out.setdefault(top_id, [])
            for child in category.get("children", []):
                sub_name = child["name"]
                for task in child.get("tasks", []):
                    task_name = task["name"]
                    for detail in task.get("details_zh", []):
                        out[top_id].append({
                            "top_id": top_id,
                            "category_name": top_name,
                            "sub_category": sub_name,
                            "task": task_name,
                            "detail": detail,
                        })
        return out
    except Exception as e:
        print(f"Failed to parse taxonomy JSON: {e}")
        return None


def extract_diverse_concepts(categorized_data, count=16):
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

def encode_image_from_bytes(img_bytes, target_area=1024 * 1024):
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            w, h = img.size
            current_area = w * h
            if current_area != target_area:
                scale = (target_area / current_area) ** 0.5
                img = img.resize(
                    (int(round(w * scale)), int(round(h * scale))),
                    Image.Resampling.LANCZOS,
                )
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def load_image_bytes(image_ref, source_type, storage_auth, storage_mod, storage_endpoint):
    if source_type == "local":
        with open(image_ref, "rb") as f:
            return f.read()
    if source_type == "object_storage":
        bucket_name, object_key = parse_storage_path(image_ref)
        if not bucket_name:
            return None
        bucket = storage_mod.Bucket(storage_auth, storage_endpoint, bucket_name)
        return bucket.get_object(object_key).read()
    return None


# ---- Per-image worker ------------------------------------------------------

def process_one(task, categorized_db, storage_auth, storage_mod, storage_endpoint,
                save_dir, candidate_count, max_retries, batch_size):
    line_idx = task["line_idx"]
    img_idx = task["img_idx"]
    image_ref = task["image_ref"]
    source_type = task["source_type"]
    task_id = f"L{line_idx}-I{img_idx}"

    try:
        img_bytes = load_image_bytes(image_ref, source_type, storage_auth, storage_mod, storage_endpoint)
        if not img_bytes:
            return False, 0, f"{task_id} image load failed"

        img_b64 = encode_image_from_bytes(img_bytes)
        if not img_b64:
            return False, 0, f"{task_id} encode failed"

        last_error = "unknown"
        for _attempt in range(max_retries):
            concepts = extract_diverse_concepts(categorized_db, count=candidate_count)
            result_list = generate_multi_edit_data(concepts, img_b64)

            if isinstance(result_list, list) and len(result_list) > 0:
                saved_count = 0
                batch_folder = os.path.join(save_dir, f"batch_{line_idx // batch_size}")
                os.makedirs(batch_folder, exist_ok=True)

                for item in result_list:
                    selected_ids = item.get("selected_option_ids", [])
                    if not isinstance(selected_ids, list) or len(selected_ids) == 0:
                        continue
                    valid_ids = [str(v) for v in selected_ids if isinstance(v, (int, str))]
                    if not valid_ids:
                        continue
                    suffix = "_".join(valid_ids)

                    if source_type == "object_storage":
                        item["storage_image_path"] = image_ref
                    else:
                        item["local_image_path"] = image_ref
                    item["source_info"] = {"line_idx": line_idx, "img_idx": img_idx}

                    json_filename = f"{line_idx}_{img_idx}_{suffix}.json"
                    save_path = os.path.join(batch_folder, json_filename)
                    try:
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(item, f, indent=2, ensure_ascii=False)
                        saved_count += 1
                    except Exception as e:
                        last_error = f"save failed: {e}"

                if saved_count > 0:
                    return True, saved_count, "ok"
                last_error = "API returned data but no valid ids"
                continue

            if isinstance(result_list, dict) and "error" in result_list:
                err = result_list["error"]
                if "Connection error" in err:
                    time.sleep(30)
                else:
                    time.sleep(1)
                last_error = f"API error: {err}"
                continue

            last_error = "API did not return a list"

        return False, 0, f"{task_id} retries exhausted: {last_error}"

    except Exception as e:
        return False, 0, f"{task_id} exception: {e}"


# ---- Task discovery --------------------------------------------------------

def collect_tasks_from_image_dir(image_dir):
    files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return [
        {
            "line_idx": idx,
            "img_idx": 0,
            "image_ref": os.path.join(image_dir, f),
            "source_type": "local",
        }
        for idx, f in enumerate(files)
    ]


def collect_tasks_from_jsonl(jsonl_path):
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for l_idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                imgs = data.get("images", [])
                if isinstance(imgs, str):
                    imgs = json.loads(imgs)
                for i_idx, img_item in enumerate(imgs):
                    path = img_item.get("image_path")
                    if not path:
                        continue
                    src = "object_storage" if urlparse(path).scheme in ("storage", "http", "https") else "local"
                    tasks.append({
                        "line_idx": l_idx,
                        "img_idx": i_idx,
                        "image_ref": path,
                        "source_type": src,
                    })
            except Exception:
                continue
    return tasks


# ---- Main -----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate multi-concept edit instructions")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-dir", help="Directory of local images")
    src.add_argument("--jsonl", help="JSONL file with .images[].image_path entries")
    p.add_argument("--taxonomy", required=True, help="Path to taxonomy JSON")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading taxonomy...")
    concept_db = load_concept_data_hierarchical(args.taxonomy)
    if not concept_db:
        return

    print("Collecting tasks...")
    if args.image_dir:
        all_tasks = collect_tasks_from_image_dir(args.image_dir)
    else:
        all_tasks = collect_tasks_from_jsonl(args.jsonl)

    needs_oss = any(t["source_type"] == "object_storage" for t in all_tasks)
    storage_mod = storage_auth = storage_endpoint = None
    if needs_oss:
        storage_mod, ak, sk, storage_endpoint = _try_import_object_storage()
        if storage_mod is None:
            print("object storage sources requested but object_storage_client / config not available.")
            return
        storage_auth = storage_mod.Auth(ak, sk)

    if not args.no_resume:
        print("Resume scan...")
        pending = []
        dir_cache = {}
        for task in tqdm(all_tasks, desc="resume"):
            b_idx = task["line_idx"] // args.batch_size
            b_dir = os.path.join(args.save_dir, f"batch_{b_idx}")
            if b_idx not in dir_cache:
                dir_cache[b_idx] = set(os.listdir(b_dir)) if os.path.exists(b_dir) else set()
            prefix = f"{task['line_idx']}_{task['img_idx']}_"
            if not any(f.startswith(prefix) for f in dir_cache[b_idx]):
                pending.append(task)
        print(f"Pending: {len(pending)} / {len(all_tasks)}")
        all_tasks = pending

    if not all_tasks:
        print("Nothing to do.")
        return

    print(f"Launching pool (workers={args.max_workers})...")
    success_img = fail_img = total_jsons = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_one, t, concept_db, storage_auth, storage_mod, storage_endpoint,
                args.save_dir, args.candidate_count, args.max_retries, args.batch_size,
            ): t for t in all_tasks
        }
        pbar = tqdm(concurrent.futures.as_completed(futures),
                    total=len(all_tasks), unit="img")
        for future in pbar:
            ok, count, msg = future.result()
            if ok:
                success_img += 1
                total_jsons += count
            else:
                fail_img += 1
                tqdm.write(f"❌ {msg}")
            pbar.set_postfix({"OK": success_img, "Fail": fail_img, "JSONs": total_jsons})

    print(f"\nDone. success={success_img}, failed={fail_img}, jsons={total_jsons}")


if __name__ == "__main__":
    main()
