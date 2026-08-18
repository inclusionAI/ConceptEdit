import os
import sys
import glob
import json
import base64
import io
import math
import argparse
import concurrent.futures
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

# 导入同目录下的 benchmark_prompts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_prompts

# ================= 配置 =================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BENCHMARK_BASE = os.path.join(PROJECT_ROOT, "data")
RESULT_BASE = os.path.join(PROJECT_ROOT, "results")

DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_VLM_MODEL = os.environ.get("VLM_MODEL", "gpt-4.1")

client = None
VLM_MODEL = DEFAULT_VLM_MODEL

# ================= 工具函数 =================

def get_target_size(width, height, target_area=1024*1024):
    aspect_ratio = width / height
    new_h = math.sqrt(target_area / aspect_ratio)
    new_w = new_h * aspect_ratio
    final_w = (int(new_w) // 16) * 16
    final_h = (int(new_h) // 16) * 16
    return max(final_w, 16), max(final_h, 16)

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG", optimize=False)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ================= 评测核心 =================

def call_vlm_metric(instruction, b64_orig, b64_edit, prompt_template):
    if client is None:
        return 0.0, "Error: OpenAI client is not initialized"

    full_prompt = prompt_template.replace("<instruction>", instruction)
    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_orig}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_edit}"}},
                ],
            }],
            max_tokens=4096,
            temperature=0.1,
            timeout=180
        )
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        res = json.loads(content)
        return float(res.get("score", 0)), res.get("reason", "")
    except Exception as e:
        return 0.0, f"Error: {str(e)}"

def evaluate_sample(task):
    try:
        orig_img = Image.open(task['orig_path']).convert("RGB")
        edit_img = Image.open(task['edited_path']).convert("RGB")

        tw, th = get_target_size(orig_img.width, orig_img.height)
        orig_img = orig_img.resize((tw, th), Image.Resampling.LANCZOS)
        edit_img = edit_img.resize((tw, th), Image.Resampling.LANCZOS)

        b64_orig = encode_image(orig_img)
        b64_edit = encode_image(edit_img)

        metric_configs = {
            "if": benchmark_prompts._prompts_if,
            "vq": benchmark_prompts._prompts_vq,
            "nc": benchmark_prompts._prompts_nc
        }

        score_out = {}
        full_results = {}
        for m_key, template in metric_configs.items():
            score, reason = call_vlm_metric(task['prompt'], b64_orig, b64_edit, template)
            score_out[m_key] = score
            full_results[f"{m_key}_score"] = score
            full_results[f"{m_key}_reason"] = reason
        return task, score_out, full_results
    except Exception as e:
        return task, {"if": 0, "nc": 0, "vq": 0}, {"error": str(e)}

# ================= 统计 =================

class HierarchicalScorer:
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.records = []

    def add(self, info, scores, full):
        valid_k = [k for k in ['if', 'nc', 'vq'] if k in scores]
        avg = sum([scores[k] for k in valid_k]) / max(1, len(valid_k))
        self.data[info['cat']][info['subcat']][info['task_name']][info['detail']].append(avg)
        self.records.append({"info": info, "metrics": scores, "full": full, "avg": avg})

    def get_report(self):
        report = {"categories": {}, "total_avg": 0}
        all_avg = []
        for cat, subcats in self.data.items():
            report["categories"][cat] = {"sub_categories": {}}
            for sub, tasks in subcats.items():
                report["categories"][cat]["sub_categories"][sub] = {"tasks": {}}
                for task, details in tasks.items():
                    report["categories"][cat]["sub_categories"][sub]["tasks"][task] = {"details": details}

        for cat_data in report["categories"].values():
            for sub_data in cat_data["sub_categories"].values():
                for task_data in sub_data["tasks"].values():
                    for detail_scores in task_data["details"].values():
                        all_avg.extend(detail_scores)

        report["total_avg"] = sum(all_avg) / len(all_avg) if all_avg else 0
        return report

# ================= 主程序 =================

def main():
    parser = argparse.ArgumentParser(description="ConceptBench 评测 (VLM Judge)")
    parser.add_argument("--model-name", type=str, required=True, help="模型名称（对应 results/ 下的目录名）")
    parser.add_argument("--threads", type=int, default=20, help="并发线程数")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="OpenAI-compatible API key; defaults to OPENAI_API_KEY")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL; defaults to OPENAI_BASE_URL or https://api.openai.com/v1")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_VLM_MODEL, help="VLM judge model; defaults to VLM_MODEL or gpt-4.1")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    global client, VLM_MODEL
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    VLM_MODEL = args.judge_model

    scorer = HierarchicalScorer()
    json_paths = glob.glob(os.path.join(BENCHMARK_BASE, "*", "*", "*", "*.json"))

    tasks_to_eval = []
    skipped = 0
    for jp in json_paths:
        rel = os.path.relpath(jp, BENCHMARK_BASE).split(os.sep)
        if len(rel) != 4:
            continue
        cat, sub, task, detail_file = rel
        detail = os.path.splitext(detail_file)[0]

        img_p = os.path.join(RESULT_BASE, args.model_name, cat, sub, task, f"{detail}.png")

        with open(jp, 'r') as f:
            data = json.load(f)

        orig_path = data.get("local_image_path", "")
        if orig_path and not os.path.isabs(orig_path):
            orig_path = os.path.join(BENCHMARK_BASE, orig_path)

        t_info = {
            "cat": cat,
            "subcat": sub,
            "task_name": task,
            "detail": detail,
            "prompt": data.get("detailed_instruction_en") or data.get("instruction_en"),
            "orig_path": orig_path,
            "edited_path": img_p
        }

        if not os.path.exists(img_p):
            scorer.add(t_info, {"if": 0, "nc": 0, "vq": 0}, {"error": "file_not_found"})
            skipped += 1
        elif not os.path.exists(orig_path):
            scorer.add(t_info, {"if": 0, "nc": 0, "vq": 0}, {"error": "orig_not_found"})
            skipped += 1
        else:
            tasks_to_eval.append(t_info)

    print(f"启动评测: {len(tasks_to_eval)} 个任务, {args.threads} 线程 (跳过 {skipped} 个)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        for t, s, f in tqdm(executor.map(evaluate_sample, tasks_to_eval), total=len(tasks_to_eval), desc="Evaluation"):
            scorer.add(t, s, f)

    final_report = scorer.get_report()
    final_report["detailed_samples"] = scorer.records
    out_dir = os.path.join(RESULT_BASE, args.model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "benchmark_report.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"\n评测完成！总平均分: {final_report['total_avg']:.2f}")
    print(f"报告已保存至: {out_file}")

if __name__ == "__main__":
    main()
