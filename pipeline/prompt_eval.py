"""
System / user prompt builder for VQA-based edit evaluation.

Used by ``eval_metric.py`` (single-concept) and ``multi_eval_metric.py``
(multi-concept). The evaluator model receives the original image, the
edited image and a set of binary (yes/no) questions and is asked to:

  1. judge each question independently;
  2. decide whether the edit is worth keeping;
  3. optionally produce a "recaption" instruction that better describes
     the actual visual change.
"""


def get_eval_prompts(concept_text, instruction_en, instruction_zh,
                     questions_text, orig_b64, edit_b64):
    """Return (system_prompt, user_content) for the evaluator chat call."""

    system_prompt = (
        "You are an expert image editing evaluator and dataset curator. "
        "Your goal is to identify high-quality image pairs (Original vs. Edited) and ensure the instructions perfectly describe the transformation.\n\n"
        "### EVALUATION CATEGORIES:\n"
        "For each question, do NOT answer 'yes' or 'no'. Instead, provide a 'quality_judgment':\n"
        "- 'positive': This dimension is high-quality. (e.g., Target edited correctly, background/non-edited preserved perfectly, NO artifacts found, IGNORING reasonable physical interactions caused by the edit, such as lighting, shadows, reflections, occlusion, or other effects introduced by the new object, etc.).\n"
        "- 'negative': This dimension is low-quality. (e.g., Target edit failed, background changed accidentally, artifacts/blur/distortions found).\n\n"
        "### FINAL DECISION PHILOSOPHY (RECAPTION TO RESCUE):\n"
        "We want to keep any image pair where the transformation is visually high-quality, even if it failed the original instruction. If the image is good, we 'fix' the data by writing a new instruction.\n\n"
        "- KEEP (`keep`: true) IF:\n"
        "  1. The edit is visually realistic and high-quality.\n"
        "  2. The changes between Original and Edited are clear and logical (even if they differ from the original instruction).\n"
        "  3. Non-edited areas are well-preserved.\n"
        "  *Action*: If it deviates from the original instruction but is still a good edit, you MUST write a new `recaption_prompt` that describes the ACTUAL transformation.\n\n"
        "- DISCARD (`keep`: false) IF:\n"
        "  1. NO visible changes occurred (images are identical).\n"
        "  2. The edit introduced severe artifacts, blurring, or 'AI-generated' messiness.\n"
        "  3. The logic is broken (e.g., an object is floating without shadows, or perspective is wrong).\n"
        "  4. The original image was already of poor quality.\n"
        "  5. Global quality degradation or prominent unintended shifts occurred. (e.g., The WHOLE image suffered from severe overexposure, oversaturation, or unnatural tone shifts; or unintended camera changes like incorrect focus shifts).\n\n"
        "### CRITICAL RULES:\n"
        "1. REASON FIRST: Analyze the visual evidence in 'reason' before providing 'quality_judgment'.\n"
        "2. VALUE ALIGNMENT: 'positive' means the image passed the quality check for that dimension. If unedited areas are consistent (identical or logically adapted), that is a 'positive' result.\n"
        "3. **DISTINGUISH INFLUENCE VS. ERROR**:\n"
        "   - **ACCEPT** reasonable interactions: e.g., if you add a lamp, the wall becoming brighter is CORRECT. If you add a red ball, a red reflection on the floor is CORRECT.\n"
        "   - **REJECT** global degradation: e.g., if you edit a cup, but the distant mountains change contrast, color tone, or become blurry, this is an ERROR.\n"
        "4. Analyze the image carefully. Do not hallucinate changes that are not there.\n"
        "5. Output strictly in valid JSON format."
    )

    user_content = [
        {
            "type": "text",
            "text": f"""
**Task:** Evaluate the image editing result.

**Edit Concept Information:**
{concept_text}

**Original Instruction (EN):** {instruction_en}
**Original Instruction (ZH):** {instruction_zh}

**Questions:**
{questions_text}

**Instructions for JSON Output:**
1. Compare Edited vs. Original Image based on the dimensions.
2. For `final_result.keep`:
   - Set to `true` if the edit is high-quality and logical, regardless of whether it matches the original instruction.
   - Set to `false` if there are artifacts, no change, or ruined quality.
3. For `final_result.recaption_prompt_en` and `final_result.recaption_prompt_zh`:
   - **Crucial Rule:** If `keep` is `true` AND the Original Instruction (English or Chinese) already **perfectly and accurately** describes the actual visual change, set both fields to \"\".
   - If `keep` is `true` BUT the Original Instruction is **not accurate or missing details**, provide a new, precise instruction in both English and Chinese.
   - If `keep` is `false`, set both to \"\".

**Target JSON Format (Strictly Follow This Structure):**
{{
  "answers": [
    {{
      "question_index": 1,
      "reason": "The dragon's horns are still golden and identical to the original image.",
      "quality_judgment": "positive"
    }},
    {{
      "question_index": 2,
      "reason": "Reason for question 2...",
      "quality_judgment": "positive"
    }}
  ],
  "final_result": {{
    "reason": "Overall analysis combining the edit success, artifacts, and preservation...",
    "keep": true,
    "recaption_prompt_en": "A new English instruction only if the original was inaccurate.",
    "recaption_prompt_zh": "对应的中文指令，仅当原指令不准确时提供。"
  }}
}}
""",
        },
        {"type": "text", "text": "Image 1: Original Image"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
        {"type": "text", "text": "Image 2: Edited Image"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{edit_b64}"}},
    ]

    return system_prompt, user_content
