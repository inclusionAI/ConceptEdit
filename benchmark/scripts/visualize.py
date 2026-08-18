import gradio as gr
import json
import os
import numpy as np

# ================= 配置区 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TAXONOMY_PATH = os.path.join(PROJECT_ROOT, "data", "taxonomy.json")
REPORT_PATH = None  # 通过命令行参数指定

def load_score_map():
    """解析报告，构建 Detail -> {avg, if, vq, nc} 映射"""
    if not os.path.exists(REPORT_PATH): return {}
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    score_map = {}
    for s in data.get("detailed_samples", []):
        info = s["info"]
        m = s["metrics"]
        # 确保所有基础数值都保留两位小数
        score_map[info["detail"]] = {
            "avg": round(float(s.get("avg", 0)), 2),
            "if": round(float(m.get("if", 0)), 2),
            "vq": round(float(m.get("vq", 0)), 2),
            "nc": round(float(m.get("nc", 0)), 2)
        }
    return score_map

def calc_stats(detail_list, score_map):
    """计算一组 Details 的平均表现，确保结果保留两位小数"""
    scores = [score_map[d] for d in detail_list if d in score_map]
    if not scores: return {"avg": 0.00, "if": 0.00, "vq": 0.00, "nc": 0.00}
    return {
        "avg": round(np.mean([s["avg"] for s in scores]), 2),
        "if": round(np.mean([s["if"] for s in scores]), 2),
        "vq": round(np.mean([s["vq"] for s in scores]), 2),
        "nc": round(np.mean([s["nc"] for s in scores]), 2)
    }

def render_score_badge(stats):
    """渲染层级旁边的均分标签"""
    score = stats['avg']
    color = "#52c41a" if score > 8 else ("#faad14" if score > 6 else "#f5222d")
    return f"""
    <span style='background:#f0f7ff; border:1px solid #91d5ff; border-radius:12px; padding:2px 8px; font-size:11px; margin-left:10px;'>
        <b style='color:{color}'>{score:.2f}</b> <span style='color:#666'>[I:{stats['if']:.2f} V:{stats['vq']:.2f} N:{stats['nc']:.2f}]</span>
    </span>"""

def build_ui():
    with open(TAXONOMY_PATH, 'r', encoding='utf-8') as f:
        tax_data = json.load(f)
    score_map = load_score_map()

    with gr.Blocks(title="Benchmark Score Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 任务全景评测图谱")
        
        for cat in tax_data.get("categories", []):
            cat_details = [d for sub in cat.get("children", []) for t in sub.get("tasks", []) for d in t.get("details_zh", [])]
            cat_stats = calc_stats(cat_details, score_map)
            
            with gr.Column(variant="panel"):
                gr.HTML(f"<h2 style='color:#10239e;'>{cat['name']} {render_score_badge(cat_stats)}</h2>")
                
                for sub in cat.get("children", []):
                    sub_details = [d for t in sub.get("tasks", []) for d in t.get("details_zh", [])]
                    sub_stats = calc_stats(sub_details, score_map)
                    
                    with gr.Accordion(label=f"📂 {sub['name']} {render_score_badge(sub_stats)}", open=True):
                        for task in sub.get("tasks", []):
                            task_stats = calc_stats(task.get("details_zh", []), score_map)
                            
                            with gr.Group():
                                gr.HTML(f"<div style='margin:10px 0;'><b>📌 {task['name']}</b> {render_score_badge(task_stats)}</div>")
                                
                                html = f"<div style='display:flex; flex-wrap:wrap; gap:8px; margin-bottom:15px;'>"
                                for d in task.get("details_zh", []):
                                    m = score_map.get(d)
                                    if m:
                                        c = "#52c41a" if m['avg'] > 8 else ("#faad14" if m['avg'] > 6 else "#f5222d")
                                        html += f"""
                                        <span style='background:#ffffff; border:1px solid {c}; border-radius:12px; padding:2px 10px; font-size:12px;'>
                                            <b>{d}</b>: <span style='color:{c}; font-weight:bold'>{m['avg']:.2f}</span>
                                            <span style='color:#999; font-size:10px;'>[I:{m['if']:.2f} V:{m['vq']:.2f} N:{m['nc']:.2f}]</span>
                                        </span>"""
                                html += "</div>"
                                gr.HTML(html)
                                gr.HTML("<div style='height:1px; background:#eee;'></div>")
    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="模型名称")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    REPORT_PATH = os.path.join(PROJECT_ROOT, "results", args.model_name, "benchmark_report.json")
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=args.port)