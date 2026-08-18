"""
Multi-concept instruction generator.

Like ``prompt_single`` but each generated instruction set combines 2-5
distinct edits that target different objects / regions of the image.
"""

import json
import re
import threading
from openai import OpenAI

from config import (
    VLM_MULTI_BASE_URL as BASE_URL,
    VLM_MULTI_API_KEY as API_KEY,
    VLM_MULTI_MODEL as MODEL_NAME,
)

# Thread-local OpenAI clients (one per worker thread)
_thread_local = threading.local()


def get_thread_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            max_retries=2,
            timeout=120.0,
        )
    return _thread_local.client


def generate_multi_edit_data(concepts_list, base64_image):
    """Build the prompt, call the VLM, return a list of multi-edit JSONs.

    Each returned dict groups 2-5 parallel edits with a unified
    instruction (en/zh) and a VQA test set.
    """
    client = get_thread_client()

    concept_text_block = ""
    for idx, c in enumerate(concepts_list):
        concept_text_block += f"""[Option ID: {idx}]
- Category: {c['category_name']} (Taxonomy ID: {c['top_id']})
- Task: {c['sub_category']} -> {c['task']}
- Detail: {c['detail']}
"""

    prompt_text = f"""
You are an expert Vision-Language AI dataset annotator.
I will provide an image and {len(concepts_list)} "Candidate Concepts".

### YOUR CORE MISSION:
Identify candidate concepts and group them into **MULTIPLE independent editing instruction sets**. Return a JSON array containing these sets.

1. **Goal**: Find as many high-quality editing opportunities as possible. Each instruction set MUST combine exactly **2 to 5 distinct parallel edits**. Aim to generate **2 to 3 distinct JSON objects (sets)** if the image complexity allows.
2. **Quality Over Quota (The "Balance" Rule)**: While rich combinations are the target, **DO NOT force nonsensical edits**. Quality, logical harmony, and visual common sense must always prevail.
3. **Quantity Target**: Strive to generate 3 independent sets by exploring all parts of the image, provided each set remains logical and natural.
4. **CONCEPT EXCLUSIVITY (Minimize Overlap)**:
   - DO NOT repeat the exact same editing action across different sets (e.g., do not "add a cat" in Set A and "add a cat" in Set B).
   - **Avoid Region Overlap**: Highly prefer targeting different objects or regions across different sets. For example, if Set A heavily edits the woman's face and shirt, Set B should ideally focus on the background, the table, or other unedited objects. Every set should try to act on independent parts of the image.

### CANDIDATE CONCEPTS:
{concept_text_block}

**HIGHLY RECOMMENDED: SPATIAL & DEPTH DISTRIBUTION WITHIN A SET**
You should aim to distribute your edits as evenly as possible across the entire image within each set, depending on the image's composition.
- **Spatial Grid**: Try to cover different areas (Top, Bottom, Left, Right, Center) if the layout allows.
- **Depth Planes**: It is recommended to involve different depths (Foreground, Midground, Background) to create a richer multi-layer edit.
- **Avoid Clustering**: DO NOT cluster all edits tightly on a single object if there is room elsewhere.

### PART 1: SELECTION & TAXONOMY CONSTRAINTS (Hard Rules)
1. **Parallel over Serial (CRITICAL)**: Prioritize edits that operate independently in parallel across different regions. DO NOT select sequential or nested edits (e.g., DO NOT "add a handbag" and then "add a charm to the handbag").
2. **NO Global/Style Modifications (FORBIDDEN)**: You MUST REJECT any purely global modifications (e.g., overall filter, global lighting, weather change, or image style transfer).
3. **Background Replacements (LAST RESORT ONLY)**: Localized foreground edits are strictly preferred. ONLY use background replacement if you have absolutely no other choice to reach the minimum concept count.
4. **Relevance & Existence**: The concept MUST target an object or attribute that *clearly exists* in the image. If the target is missing, REJECT it.
5. **"portrait_human_specialized" (Portrait & Body)**:
   - IF the image has NO clear human face or body -> REJECT.
   - Close-up faces -> ACCEPT micro-edits, REJECT full-body edits.
6. **Combination Appropriateness**: Ensure the selected concepts do not logically conflict.

### PART 2: MANDATORY FALLBACK LOGIC (Options 90 to 99)
If the provided candidates are exhausted and you cannot form enough sets of 2-5 edits, you are **STRICTLY REQUIRED** to actively scan the image for untapped potential (empty space, clutter, objects to modify) and invent additional concepts.
- Use **Option IDs 90, 91, 92, etc.** for invented concepts.
Select their `edit_concept` EXACTLY from the following templates:
- **Type A (Universal Add)**: {{"category": "通用物体与实体编辑", "sub_category": "物体管理与操作", "task": "增删目标", "detail": "通用添加"}}
- **Type B (Universal Remove)**: {{"category": "通用物体与实体编辑", "sub_category": "物体管理与操作", "task": "增删目标", "detail": "通用删除"}}
- **Type C (Universal Modify)**: {{"category": "通用物体与实体编辑", "sub_category": "物体管理与操作", "task": "替换目标", "detail": "通用替换和更改"}}

### PART 3: NO CHINESE TEXT ALTERATION (CRITICAL REJECTION RULE)
**REJECT** any concept that generates, alters, or re-renders the **SHAPE/STROKES of Chinese characters (中文汉字)** (e.g., adding new Chinese text, altering text content, or changing fonts).
**ACCEPT**: Replacing Chinese with other languages, changing text color, purely erasing text, or global/general edits that do not alter text shapes.

### PART 4: INSTRUCTION WRITING GUIDELINES (Context-Aware & Short Sentences)
For each instruction set, write ONE combined instruction pair (`instruction`, `detailed_instruction`) in **BOTH English and Chinese**.
1. **Command Style**: The instructions MUST be **EDITING COMMANDS (祈使句/指令)**. DO NOT describe the final image.
2. **Context-Aware**: Do not simply repeat the taxonomy 'detail'. Fuse the concept with the specific visual content.
3. **Short Fluent Sentences**: The instructions should be written as fluent short sentences. Ideally, **each sentence corresponds to one edit**.
   - *Bad*: "Change hair color, add glasses, and add a cup."
   - *Good*: "Change the woman's hair to blonde. Make her wear black glasses. Add a fluffy orange cat sitting on the wooden desk."

### PART 5: MULTI-CONCEPT VQA VERIFICATION
You must generate exactly **N + 4** Questions in `evaluation_vqa`, where **N** is the total number of concepts used within this specific instruction set.
**ALL questions must be strictly binary (Yes/No).** DO NOT use generic terms like "the object", use SPECIFIC nouns.

- **[Generate N Questions] 1_primary_edit_success_option_[id]**: One separate question for EACH selected concept. `expected_answer`: "yes".
- **[1 Question] 2_multi_edit_harmony**: Check the global harmony of the parallel edits. `expected_answer`: "yes".
- **[1 Question] 3_identity_and_texture_details**: Focus on fragile details. Are facial identities preserved naturally? Do objects look realistic? `expected_answer`: "yes".
- **[Generate 1 Question] 4_specific_preservation**:
  - You MUST ask if specific UNRELATED items CHANGED or GOT DAMAGED.
  - EN: "Did the [specific unrelated background/objects] change, distort, or disappear (excluding reasonable effects caused by the edits)?"
  - ZH: "[具体的无关背景/物体]是否发生了改变、扭曲或消失（忽略因编辑任务带来的合理变化）？"
  - `expected_answer`: "no"
- **[Generate 1 Question] 5_artifacts_anatomy_and_text**:
  - Detect severe multi-modal flaws.
  - EN: "Are there any visible AI artifacts, unnatural blurs, anatomical distortions in the faces/bodies, or illegible/fake text in the edited areas?"
  - ZH: "在编辑区域是否存在任何可见的AI伪影、生成人脸/人体的解剖结构崩坏扭曲，或者是无法阅读的虚假乱码文字？"
  - `expected_answer`: "no"

### OUTPUT FORMAT (Strict JSON Array):
Return ONLY a JSON Array `[...]` containing multiple dictionary objects.
Below is a full valid example demonstrating how to structure the outputs.
The second dictionary is abbreviated for demonstration but should be fully generated in your output.

[
  {{
    "step_1_spatial_and_depth_analysis": "The image features a wooden table at the bottom center, a young woman in a blue shirt in the middle, and a blurred street with a cafe window at the top.",
    "step_2_logical_deduction": "Using Option 0 (Change hair color), Option 2 (Add glasses), and inventing Option 90 (Universal modify). N=3. Total VQA = 7.",
    "selected_option_ids": [0, 2, 90],
    "caption": "A young woman wearing a blue shirt is sitting at a wooden cafe table.",
    "edit_concepts_used": [
      {{ "option_id": 0, "category": "人像及人体专业编辑", "sub_category": "人像属性修改", "task": "修改发色", "detail": "将头发改为粉色" }},
      {{ "option_id": 2, "category": "人像及人体专业编辑", "sub_category": "人像配饰修改", "task": "添加配饰", "detail": "添加眼镜" }},
      {{ "option_id": 90, "category": "通用物体与实体编辑", "sub_category": "物体管理与操作", "task": "替换目标", "detail": "通用替换和更改" }}
    ],
    "instruction_en": "Change the woman's hair to vibrant pink. Make her wear black square glasses. Change her blue shirt to a white t-shirt.",
    "instruction_zh": "把女人的头发变成亮粉色。让她戴上黑色的黑框眼镜。把她的蓝色衬衫换成白色T恤。",
    "detailed_instruction_en": "Change the young woman's hair color to a vibrant pink. Place black square-framed glasses on her face. Change her blue shirt into a crisp white t-shirt.",
    "detailed_instruction_zh": "将年轻女人的头发颜色改成亮粉色。在她的脸上加上一副黑色的方形边框眼镜。把她的蓝色衬衫变成一件干净的白色T恤。",
    "evaluation_vqa": [
      {{ "dimension": "1_primary_edit_success_option_0", "question_en": "Is the young woman's hair now vibrant pink?", "question_zh": "年轻女人的头发现在是亮粉色吗？", "expected_answer": "yes" }},
      {{ "dimension": "1_primary_edit_success_option_2", "question_en": "Is the young woman now wearing black square glasses?", "question_zh": "年轻女人现在是否戴着黑色的方形眼镜？", "expected_answer": "yes" }},
      {{ "dimension": "1_primary_edit_success_option_90", "question_en": "Is the young woman now wearing a crisp white t-shirt instead of a blue shirt?", "question_zh": "年轻女人现在是否穿着干净的白色T恤而不是蓝色衬衫？", "expected_answer": "yes" }},
      {{ "dimension": "2_multi_edit_harmony", "question_en": "Do the newly edited pink hair, glasses, and white shirt integrate naturally with the scene's lighting?", "question_zh": "新编辑的粉色头发、眼镜和白衬衫是否与场景的光照自然融合？", "expected_answer": "yes" }},
      {{ "dimension": "3_identity_and_texture_details", "question_en": "Is the woman's original facial identity preserved under the glasses?", "question_zh": "戴上眼镜后女人的原始面部特征是否得以保留？", "expected_answer": "yes" }},
      {{ "dimension": "4_specific_preservation", "question_en": "Did the wooden table or the background street change in any way?", "question_zh": "木桌或背景的街道是否发生了任何改变？", "expected_answer": "no" }},
      {{ "dimension": "5_artifacts_anatomy_and_text", "question_en": "Are there any anatomical distortions in the woman's face or body?", "question_zh": "女人的面部或身体是否存在任何解剖结构扭曲？", "expected_answer": "no" }}
    ]
  }}
]
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            max_tokens=16384,
            temperature=0.7,
            top_p=0.8,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        raw_content = response.choices[0].message.content.strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content, flags=re.IGNORECASE)
            raw_content = re.sub(r"\s*```$", "", raw_content)
            raw_content = raw_content.strip()

        parsed = json.loads(raw_content)
        return [parsed] if isinstance(parsed, dict) else parsed

    except json.JSONDecodeError as e:
        return {"error": f"JSON Decode failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Process failed: {str(e)}"}
