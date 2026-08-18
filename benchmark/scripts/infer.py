import os
import glob
import json
import torch
import torch.multiprocessing as mp
import math
import concurrent.futures
import argparse
from PIL import Image
from tqdm import tqdm
from diffusers import Flux2KleinPipeline

# ================= 配置 =================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BENCHMARK_BASE = os.path.join(PROJECT_ROOT, "data")
RESULT_BASE = os.path.join(PROJECT_ROOT, "results")

DTYPE = torch.bfloat16

# ================= 辅助函数 =================

def get_target_size(width, height, target_area=1024*1024):
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
        result_queue.put({"status": "error", "msg": f"[GPU {rank}] Save Error: {e}"})

# ================= GPU Worker =================

def gpu_worker(rank, task_queue, result_queue, model_path):
    try:
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

        pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=DTYPE)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        generator = torch.Generator(device=device).manual_seed(42)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                task = task_queue.get()
                if task is None:
                    break

                img_path = task['img_path']
                prompt = task['prompt']
                save_path = task['save_path']

                try:
                    base_image = Image.open(img_path).convert("RGB")
                    orig_w, orig_h = base_image.size
                    target_w, target_h = get_target_size(orig_w, orig_h)
                    base_image_resized = base_image.resize((target_w, target_h), Image.Resampling.LANCZOS)

                    result = pipe(
                        image=base_image_resized,
                        prompt=prompt,
                        guidance_scale=1.0,
                        num_inference_steps=4,
                        generator=generator
                    ).images[0]

                    executor.submit(async_save_image, result, save_path, result_queue, rank)

                except Exception as e:
                    result_queue.put({"status": "error", "msg": f"[GPU {rank}] Failed {os.path.basename(img_path)}: {e}"})

    except Exception as e:
        print(f"[GPU {rank}] Critical crash: {e}")
        result_queue.put({"status": "error", "msg": f"[GPU {rank}] Critical Crash"})

# ================= 主程序 =================

def main():
    parser = argparse.ArgumentParser(description="ConceptBench 推理 (Flux2Klein)")
    parser.add_argument("--model-name", type=str, required=True, help="模型名称，用于保存路径")
    parser.add_argument("--model-path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--num-gpus", type=int, default=8, help="使用的 GPU 数量")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    print(f"正在扫描 {BENCHMARK_BASE} ...")
    all_jsons = glob.glob(os.path.join(BENCHMARK_BASE, "*", "*", "*", "*.json"))

    tasks = []
    skipped = 0

    for json_path in all_jsons:
        try:
            rel_path = os.path.relpath(json_path, BENCHMARK_BASE)
            base_name = os.path.splitext(rel_path)[0]
            save_path = os.path.join(RESULT_BASE, args.model_name, f"{base_name}.png")

            if os.path.exists(save_path):
                skipped += 1
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            prompt = data.get("detailed_instruction_en") or data.get("instruction_en")
            img_path = data.get("local_image_path")
            # 支持相对路径
            if img_path and not os.path.isabs(img_path):
                img_path = os.path.join(BENCHMARK_BASE, img_path)

            if img_path and prompt and os.path.exists(img_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                tasks.append({
                    "json_path": json_path,
                    "img_path": img_path,
                    "prompt": prompt,
                    "save_path": save_path
                })
        except Exception as e:
            print(f"解析 {json_path} 时出错: {e}")

    print(f"\n任务汇总: 总 {len(all_jsons)} | 待推理 {len(tasks)} | 跳过已完成 {skipped}\n")

    if not tasks:
        print("没有可执行的新任务。")
        return

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for task in tasks:
        task_queue.put(task)
    for _ in range(args.num_gpus):
        task_queue.put(None)

    processes = []
    print(f"启动 {args.num_gpus} 个 GPU 推理...")
    for rank in range(args.num_gpus):
        p = mp.Process(target=gpu_worker, args=(rank, task_queue, result_queue, args.model_path))
        p.start()
        processes.append(p)

    with tqdm(total=len(tasks), desc=f"Inferencing [{args.model_name}]") as pbar:
        completed = 0
        while completed < len(tasks):
            res = result_queue.get()
            completed += 1
            pbar.update(1)
            if res["status"] == "error":
                pbar.write(res["msg"])

    for p in processes:
        p.join()

    print("\n推理完成！")

if __name__ == "__main__":
    main()
