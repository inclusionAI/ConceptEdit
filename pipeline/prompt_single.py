"""
Single-concept instruction generator.

Given an image and a small set of candidate editing concepts sampled from
a taxonomy, calls a VLM and asks it to choose the most suitable concept(s)
and emit a structured editing instruction (plus a VQA test set used later
during evaluation).
"""

import json
import threading
from openai import OpenAI

from config import (
    VLM_SINGLE_BASE_URL as BASE_URL,
    VLM_SINGLE_API_KEY as API_KEY,
    VLM_SINGLE_MODEL as MODEL_NAME,
)

# Each worker thread gets its own OpenAI client to avoid connection
# pool contention under high concurrency.
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
    """Build the prompt, call the VLM and return a list of edit JSONs.

    Args:
        concepts_list: list of candidate concept dicts. Each dict must
            contain ``top_id``, ``category_name``, ``sub_category``,
            ``task`` and ``detail``.
        base64_image: PNG-encoded image, base64 string (no data URI prefix).

    Returns:
        A list of structured edit dicts on success, or ``{"error": ...}``.
    """
    client = get_thread_client()

    # 1. Build candidate-concepts block
    concept_text_block = ""
    for idx, c in enumerate(concepts_list):
        concept_text_block += f"""
[Option ID: {idx}]
- Taxonomy ID: {c['top_id']}
- Category: {c['category_name']}
- Task: {c['sub_category']} -> {c['task']}
- Detail: {c['detail']}
"""

    # 2. Build full prompt
    prompt_text = f"""
You are an expert Vision-Language AI dataset annotator.
I will provide an image and {len(concepts_list)} "Candidate Concepts".

### YOUR CORE MISSION:
Identify and select the most suitable editing concepts for this image.
**Goal: Find as many high-quality editing opportunities as possible (aiming for 2 to 3 distinct options).**

1. **Selection & Expansion**: Actively scan the image for multiple valid editing paths (e.g., one portrait edit, one background change, and one object modification). **It is highly preferred to provide 2 or 3 options rather than just one, provided they are all logically grounded and suitable.**
2. **Hierarchy of Choice**:
   - First, select all suitable concepts from the **[CANDIDATE CONCEPTS]**.
   - If the provided candidates are exhausted or insufficient to reach the target count (2-3), but the image clearly has "untapped" potential (e.g., empty space, removable clutter), you are authorized to **Freely Invent** a perfect editing concept using **Option ID 99**.
3. **Quality Over Quota (The "Balance" Rule)**: While 2-3 is the target, **DO NOT force it**. If only 1 edit is truly logically sound and visually natural, stop there. Do not suggest "forced" or "nonsensical" edits just to fill the count. Quality and visual common sense must always prevail.

### CANDIDATE CONCEPTS:
{concept_text_block}

### PART 1: SELECTION LOGIC (Value Judgment)
1.  **Relevance & Existence**: The concept MUST target an object or attribute that *clearly exists* in the image. If the target is missing, REJECT it. **Do not add objects just to remove them.**
2.  **Visual Impact**: Prioritize edits that produce a **visible, significant transformation** over subtle, invisible tweaks.
3.  **Specialization Bias**:
    -   Clear Face/Body? -> Prioritize Portrait edits.
    -   Clear Text? -> Prioritize Text edits.

### PART 2: TAXONOMY CONSTRAINTS & GRANULARITY (Hard Rules)
**1. "portrait_human_specialized" (Portrait & Body)**
   - **Constraint**: IF the image has NO clear human face or body -> REJECT.
   - **Scale Check**: Close-up faces -> ACCEPT micro-edits (makeup, teeth, wrinkles), REJECT full-body (change pants, pose). Wide-shot/Crowd -> REJECT micro-edits, ACCEPT macro-edits.
**2. "text_graphic_design" & "general_object_editing"**
   - **Constraint**: Cannot add/replace objects where there is no physical space or logical context (e.g., don't add "furniture" to a sky photo).
**3. "advanced_domain_application" (E-commerce)**
   - **Constraint**: E-commerce edits (virtual try-on, product podiums) MUST apply to clear products, models, or still-life photography.

### PART 3: FALLBACK LOGIC (Universal Generic Actions)
Trigger this ONLY if all Candidate Concepts are rejected.
Set `option_id` to **99**.
Select ONE of the following 3 generic concepts that best fits the image potential, and fill the `edit_concept` block EXACTLY with the provided JSON values:

**Type A: Universal Add (If the image has empty space suitable for adding objects)**
{{
    "category": "通用物体与实体编辑",
    "sub_category": "物体管理与操作",
    "task": "增删目标",
    "detail": "通用添加"
}}

**Type B: Universal Remove (If the image has clear clutter/objects to remove)**
{{
    "category": "通用物体与实体编辑",
    "sub_category": "物体管理与操作",
    "task": "增删目标",
    "detail": "通用删除"
}}

**Type C: Universal Modify (If an object needs style/color/texture change)**
{{
    "category": "通用物体与实体编辑",
    "sub_category": "物体管理与操作",
    "task": "替换目标",
    "detail": "通用替换和更改"
}}

### PART 4: INSTRUCTION DESIGN (Context-Aware Fusion)
**CRITICAL RULE:** The `detail` field in the taxonomy is often a broad class (e.g., "Remove passerby", "Change weather", "通用添加").
Your generated `instruction` and `detailed_instruction` **MUST NOT** simply repeat the `detail` or `task` name.
You **MUST** fuse the concept with the specific visual content of the image.

*   **Bad Example**:
    *   Detail: "Remove passerby"
    *   Instruction: "Remove the passerby." (Too generic)
*   **Good Example**:
    *   Detail: "Remove passerby"
    *   Instruction: "Remove the man in the red jacket walking in the background." (Specific)
*   **Fallback Example**:
    *   Detail: "通用添加"
    *   Instruction: "Add a brown wooden chair to the empty corner on the left." (Specific)

### PART 5: CHINESE TEXT EDITING FLAG (is_chinese_text_edit)
You must determine if the editing action requires the AI model to generate, alter, or re-render the **SHAPE/STROKES of Chinese characters (中文汉字)** inside the image.
Set `"is_chinese_text_edit": true` **ONLY IF** the shape or structure of Chinese text is changed:
1. Adding NEW Chinese characters (e.g., "Add a sign saying '欢迎'").
2. Changing existing text to DIFFERENT Chinese characters (e.g., "Change '打折' to '免费'").
3. Changing the FONT or TYPOGRAPHY of existing Chinese characters (e.g., "Change the text font to handwritten style").

Set `"is_chinese_text_edit": false` for ALL other cases:
- Replacing Chinese text with English or other languages (e.g., "Change '商店' to 'Shop'").
- Merely changing text COLOR without changing its shape or font.
- Purely erasing/removing Chinese text without adding new text.
- Global image filters or style transfers (even if Chinese text is visible in the image).

### OUTPUT INSTRUCTIONS:
1. **Iterate** through all {len(concepts_list)} options.
2. **Filter**: Keep ONLY the options that pass the Strict Constraints and logical checks.
3. **Generate**: For every feasible option (up to a total of 3), generate a full detailed JSON object.
4. **Format**: Return a JSON **ARRAY** containing all feasible objects.

### OUTPUT FORMAT (Strict JSON Array):
Return ONLY a JSON Array `[...]`. The reasoning MUST be the very first keys to ensure proper Chain of Thought. Example:
[
  {{
    "step_1_image_analysis": "Brief description of the image content.",
    "step_2_logical_deduction": "Option 0 (Remove hat): Rejected, no hat. Option 2 (Cyberpunk): Accepted.",
    "option_id": 2,
    "caption": "Detailed description of the original image.",
    "edit_concept": {{
      "category": "Global Atmosphere",
      "sub_category": "Style Transfer",
      "task": "Cyberpunk",
      "detail": "Cyberpunk"
    }},
    "instruction_en": "Natural English command",
    "instruction_zh": "自然、口语化的中文指令",
    "detailed_instruction_en": "Rich descriptive instruction.",
    "detailed_instruction_zh": "极具画面细节的丰富指令描述。",
    "is_chinese_text_edit": false,
    "evaluation_vqa": [
      {{
        "dimension": "1_primary_edit_success",
        "question_en": "Is the young man in the blue shirt now wearing black wire-framed glasses?",
        "question_zh": "穿蓝衬衫的年轻男人现在是否戴着黑色金属边框眼镜？",
        "expected_answer": "yes"
      }},
      {{
        "dimension": "2_logic_and_physics",
        "question_en": "Do the temples of the glasses correctly rest on the man's ears, and is there a logical cast shadow on his cheeks?",
        "question_zh": "眼镜的镜腿是否正确且符合逻辑地架在男人的耳朵上，并且在脸颊上投射了合理的阴影？",
        "expected_answer": "yes"
      }},
      {{
        "dimension": "3_key_details",
        "question_en": "Is the man's facial identity preserved naturally, ensuring he looks like the same person despite the added glasses?",
        "question_zh": "男人的面部特征是否自然保留，确保尽管戴上了眼镜，他看起来仍然是原本那个人？",
        "expected_answer": "yes"
      }},
      {{
        "dimension": "4_specific_preservation",
        "question_en": "Did the blurred busy street in the background or the man's blue shirt change in any way (excluding reasonable effects caused by the edit)?",
        "question_zh": "背景中模糊的繁华街道或男人的蓝衬衫是否发生了任何改变（忽略因编辑任务带来的合理变化）？",
        "expected_answer": "no"
      }},
      {{
        "dimension": "5_artifacts_and_anatomy",
        "question_en": "Are there any unnatural halos, distortion, smudging, or mismatched textures in the edited area and its surroundings?",
        "question_zh": "在编辑区域及其周围是否存在任何不自然的光晕、扭曲、涂抹痕迹或纹理不匹配？",
        "expected_answer": "no"
      }}
    ]
  }}
]

### VQA REQUIREMENTS (CRITICAL FOR FINE-GRAINED FILTERING):
You MUST generate EXACTLY 5 Questions in `evaluation_vqa`.
**ANSWER PATTERN RULE: Design the questions so that Q1, Q2, and Q3 are strictly 'yes', while Q4 and Q5 are strictly 'no'.**
**CRITICAL RULE: DO NOT use generic terms like "the object", "the background". You MUST refer to SPECIFIC nouns in the image.**

- **1_primary_edit_success**: Verify the core edit using specific subjects.
  - `expected_answer`: "yes"

- **2_logic_and_physics**: Check if spatial logic, lighting/shadows, and occlusion are physically correct.
  - `expected_answer`: "yes"

- **3_key_details (Focus on Fragile Details)**:
  Check the most fragile aspect of the edit to ensure high-quality integration.
  - `expected_answer`: "yes"

- **4_specific_preservation**: You MUST ask if the items CHANGED or GOT DAMAGED. **DO NOT ask if they "remained unchanged".**
  - **Enforced Phrasing (EN)**: "Did the [specific background/unrelated items] change, distort, or disappear (excluding reasonable effects caused by the edit)?"
  - `expected_answer`: "no"

- **5_artifacts_and_anatomy (Context-Aware Check)**:
  Detect errors based on the edit type.
  - `expected_answer`: "no"

- ALL questions must be strictly binary (Yes/No).

### REQUIREMENTS:
- If you use Option 99, populate the edit_concept block using the corresponding category/sub_category/task/detail from the templates in PART 3.
- Ensure Chinese translations are native-like.
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
            max_tokens=8192,
            temperature=0.7,
            top_p=0.8,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        raw_content = response.choices[0].message.content.strip()
        # Strip optional Markdown code fences
        if raw_content.startswith("```"):
            raw_content = raw_content.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw_content)
        if isinstance(parsed, dict):
            return [parsed]
        return parsed

    except Exception as e:
        return {"error": f"Process failed: {str(e)}"}
