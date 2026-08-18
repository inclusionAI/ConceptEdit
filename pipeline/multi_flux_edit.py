"""
Step 2 (multi-concept): edit each image with FLUX using the combined
multi-edit instruction.

This is the multi-concept analogue of ``flux_edit.py``. The JSONs
produced by ``multi_instruct_gen.py`` already contain a single combined
``detailed_instruction_en`` field that bundles 2-5 parallel edits, so
inference is structurally identical: one forward pass per image.

The script is kept separate from ``flux_edit.py`` because the two
pipelines tend to evolve independently (different prompts, different
preferred guidance / step settings, etc.).
"""

import argparse
import concurrent.futures
import io
import json
import math
import os
from urllib.parse import unquote, urlparse

import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

from diffusers import Flux2KleinPipeline

from config import (
    FLUX_MODEL_PATH as MODEL_PATH,
    NUM_GPUS,
)

DTYPE = torch.bfloat16


# ---- object storage helpers (optional) ------------------------------------------------

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


# ---- Image helpers --------------------------------------------------------

def get_target_size(width, height, target_area=1024 * 1024):
    aspect_ratio = width / height
    new_h = math.sqrt(target_area / aspect_ratio)
    new_w = new_h * aspect_ratio
    final_w = (int(new_w) // 16) * 16
    final_h = (int(new_h) // 16) * 16
    return max(final_w, 16), max(final_h, 16)


def async_save_image(image, save_path, result_queue, rank):
    try:
        image.save(save_path, optimize=False, compress_level=1)
        result_queue.put({"status": "success", "file": save_path})
    except Exception as e:
        result_queue.put({"status": "error", "msg": f"[GPU {rank}] save error: {e}"})


# ---- GPU worker -----------------------------------------------------------

def gpu_worker(rank, task_queue, result_queue, model_path, dtype, seed):
    try:
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

        storage_mod, ak, sk, storage_endpoint = _try_import_object_storage()
        storage_auth = storage_mod.Auth(ak, sk) if storage_mod is not None else None

        pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        generator = torch.Generator(device=device).manual_seed(seed)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                task = task_queue.get()
                if task is None:
                    break

                json_filename = task["json_filename"]
                prompt = task["prompt"]
                save_path = task["save_path"]
                image_path = task["image_path"]
                source_type = task["source_type"]

                try:
                    if source_type == "object_storage":
                        if storage_auth is None:
                            raise RuntimeError("object storage source requested but object_storage_client unavailable")
                        bucket_name, object_key = parse_storage_path(image_path)
                        bucket = storage_mod.Bucket(storage_auth, storage_endpoint, bucket_name)
                        img_bytes = bucket.get_object(object_key).read()
                        base_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    elif source_type == "local":
                        base_image = Image.open(image_path).convert("RGB")
                    else:
                        raise ValueError(f"unknown source type: {source_type}")

                    orig_w, orig_h = base_image.size
                    target_w, target_h = get_target_size(orig_w, orig_h)
                    base_image_resized = base_image.resize((target_w, target_h),
                                                           Image.Resampling.LANCZOS)

                    result = pipe(
                        image=base_image_resized,
                        prompt=prompt,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=generator,
                    ).images[0]

                    executor.submit(async_save_image, result, save_path, result_queue, rank)

                except Exception as e:
                    result_queue.put({
                        "status": "error",
                        "msg": f"[GPU {rank}] failed {json_filename}: {e}",
                    })
    except Exception as e:
        print(f"[GPU {rank}] critical crash: {e}")


# ---- Task discovery -------------------------------------------------------

def discover_tasks(input_dirs):
    tasks = []
    skipped_chinese = skipped_resume = invalid = 0

    for target_dir in input_dirs:
        if not os.path.exists(target_dir):
            print(f"warning: {target_dir} does not exist, skipping.")
            continue

        print(f"scanning {target_dir} ...")
        json_files = [
            f for f in os.listdir(target_dir)
            if f.endswith(".json") and "_edit" not in f and "_vqa_result" not in f
        ]

        for j_file in json_files:
            json_path = os.path.join(target_dir, j_file)
            base_name = os.path.splitext(j_file)[0]
            save_path = os.path.join(target_dir, f"{base_name}_edit.png")

            if os.path.exists(save_path):
                skipped_resume += 1
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Multi-edit JSONs may occasionally be wrapped in a list
                if isinstance(data, list) and data:
                    data = data[0]

                if data.get("is_chinese_text_edit") is True:
                    skipped_chinese += 1
                    continue

                storage_path = data.get("storage_image_path")
                local_path = data.get("local_image_path")
                prompt = data.get("detailed_instruction_en")

                image_path = source_type = None
                if storage_path:
                    image_path, source_type = storage_path, "object_storage"
                elif local_path:
                    image_path, source_type = local_path, "local"

                if image_path and prompt:
                    tasks.append({
                        "json_filename": j_file,
                        "image_path": image_path,
                        "source_type": source_type,
                        "prompt": prompt,
                        "save_path": save_path,
                    })
                else:
                    invalid += 1
            except Exception:
                invalid += 1

    return tasks, skipped_resume, skipped_chinese, invalid


# ---- Main -----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Multi-concept FLUX image editing")
    p.add_argument("input_dirs", nargs="+", help="One or more directories of multi-edit JSONs")
    p.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", default=MODEL_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    tasks, skipped_resume, skipped_chinese, invalid = discover_tasks(args.input_dirs)
    total = len(tasks)
    print("\nMulti-edit summary:")
    print(f"  - new tasks                 : {total}")
    print(f"  - skipped (already done)    : {skipped_resume}")
    print(f"  - skipped (chinese text)    : {skipped_chinese}")
    print(f"  - skipped (invalid records) : {invalid}")
    if total == 0:
        return

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)
    for _ in range(args.num_gpus):
        task_queue.put(None)

    processes = []
    for rank in range(args.num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(rank, task_queue, result_queue, args.model_path, DTYPE, args.seed),
        )
        p.start()
        processes.append(p)

    with tqdm(total=total, desc="Flux Multi-Editing") as pbar:
        completed = 0
        while completed < total:
            res = result_queue.get()
            completed += 1
            pbar.update(1)
            if res["status"] == "error":
                pbar.write(res["msg"])

    for p in processes:
        p.join()
    print("\nAll done.")


if __name__ == "__main__":
    main()
