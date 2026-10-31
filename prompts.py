"""
Prompt templates used by the open-source listwise reranker.

The reranking pipeline uses non-CoT ranking prompts only: the model should
return a JSON array inside <answer> tags, containing 1-based candidate indices
from most to least relevant.
"""

import json
import re
from typing import List, Optional


SYSTEM_PROMPT_CIR = """You are an expert in Composed Image Retrieval (CIR).
Your task is to rank candidate images based on how well they match a query that combines:
1. A reference image (showing the base concept)
2. A modification text (describing desired changes)

The target image should capture the essence of the reference while incorporating the modifications."""


SYSTEM_PROMPT_TEXT2IMG = """You are an expert in text-to-image retrieval.
Your task is to rank candidate images based on how well they match the text query.
The best candidates should accurately depict the content, style, and context described in the query."""


SYSTEM_PROMPT_VISDIAL = """You are an expert in Visual Dialog-based Image Retrieval.
Your task is to retrieve the target image that best matches a multi-turn dialogue history.
Analyze the caption and the sequence of question-answer pairs to understand the visual context and specific details.
Follow the output format strictly."""


SYSTEM_PROMPT_FASHIONIQ = """You are an expert in fashion image retrieval.
Your task is to rank candidate fashion images according to a composed fashion query.

The query consists of a reference fashion image and a short modification description.
The modification description may be incomplete, comparative, or attribute-only.
Interpret the modification as changes relative to the clothing item in the reference image.

Focus on the clothing item itself: category, color, pattern, material, texture, silhouette, length, neckline, sleeves, fit, and style.
Ignore background, pose, model identity, lighting, camera angle, and image quality unless they affect the clothing item."""


USER_PROMPT_NOCOT_RANKING = """## Task
Rank the {k} candidate images from MOST to LEAST relevant to the composed query.

## Query
- Reference Image: [Image shown above]
- Modification: {query_text}

## Output
Provide ONLY the ranking as a JSON array inside answer tags:
<answer>
[best_idx, second_best_idx, ..., worst_idx]
</answer>"""


USER_PROMPT_TEXT_ONLY_NOCOT_RANKING = """## Task
Rank the {k} candidate images from MOST to LEAST relevant to the dialog query.

## Query (Dialog)
{query_text}

## Output
Provide ONLY the ranking as a JSON array inside answer tags:
<answer>
[best_idx, second_best_idx, ..., worst_idx]
</answer>"""


USER_PROMPT_TEXT2IMG_NOCOT_RANKING = """## Task
Rank the {k} candidate images from MOST to LEAST relevant to the text query.

## Query
{query_text}

## Output
Provide ONLY the ranking as a JSON array inside answer tags:
<answer>
[best_idx, second_best_idx, ..., worst_idx]
</answer>"""


USER_PROMPT_FASHIONIQ_NOCOT_RANKING_TWO_CAPTIONS = """## Task
Rank the {k} candidate fashion images from MOST to LEAST relevant to the composed fashion query.

## Query
- Reference Image: [Image shown above]
- Modification Description 1: {caption_1}
- Modification Description 2: {caption_2}

## Important Guidance
Each modification description may be short, incomplete, comparative, or attribute-only.
Interpret both descriptions as changes relative to the clothing item in the reference image.
The two descriptions may emphasize different attributes. A good candidate should satisfy both as much as possible.

## Ranking Criteria
A better candidate should:
1. Keep the same fashion item category as the reference image.
2. Preserve reference attributes unless contradicted by the modification descriptions.
3. Apply both modification descriptions as relative clothing changes.
4. Focus on fine-grained fashion attributes: color, pattern, material, texture, silhouette, length, neckline, sleeves, fit, and style.
5. Do not rank a candidate highly only because it is visually similar to the reference if it fails the requested changes.
6. Ignore background, pose, model identity, lighting, and camera angle unless they affect the clothing item.

## Output
Provide ONLY the ranking as a JSON array inside answer tags:
<answer>
[best_idx, second_best_idx, ..., worst_idx]
</answer>"""


def get_system_prompt(task_type: str = "cir") -> str:
    """Return the system prompt for a supported retrieval task."""
    if task_type == "cir":
        return SYSTEM_PROMPT_CIR
    if task_type == "fashioniq":
        return SYSTEM_PROMPT_FASHIONIQ
    if task_type == "text2img":
        return SYSTEM_PROMPT_TEXT2IMG
    if task_type == "visdial":
        return SYSTEM_PROMPT_VISDIAL
    raise ValueError(f"Unknown task type: {task_type}")


def get_user_prompt_template(
    use_cot: bool = False,
    output_format: str = "ranking",
    has_reference_image: bool = True,
    task_type: Optional[str] = None,
) -> str:
    """Return the non-CoT ranking prompt template used at inference time."""
    if use_cot:
        raise ValueError("The open-source inference prompts only support use_cot=False.")
    if output_format != "ranking":
        raise ValueError("The open-source inference prompts only support output_format='ranking'.")

    if has_reference_image:
        return USER_PROMPT_NOCOT_RANKING
    if task_type == "text2img":
        return USER_PROMPT_TEXT2IMG_NOCOT_RANKING
    return USER_PROMPT_TEXT_ONLY_NOCOT_RANKING


def format_user_prompt(
    query_text: str,
    k: int,
    use_cot: bool = False,
    output_format: str = "ranking",
    has_reference_image: bool = True,
    task_type: Optional[str] = None,
) -> str:
    """Format the user prompt for a listwise ranking call."""
    template = get_user_prompt_template(
        use_cot=use_cot,
        output_format=output_format,
        has_reference_image=has_reference_image,
        task_type=task_type,
    )
    return template.format(k=k, query_text=query_text)


def format_fashioniq_user_prompt(query_text: str, k: int) -> str:
    """Format the FashionIQ prompt from one or two comma-separated captions."""
    parts = query_text.split(", ", 1)
    caption_1 = parts[0].strip()
    caption_2 = parts[1].strip() if len(parts) > 1 else ""
    return USER_PROMPT_FASHIONIQ_NOCOT_RANKING_TWO_CAPTIONS.format(
        k=k,
        caption_1=caption_1,
        caption_2=caption_2,
    )


_ANSWER_TAG_PAT = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def parse_ranking_output(text: str, k: int) -> List[int]:
    """Parse exactly one complete 1-based permutation from a response."""
    if k <= 0:
        raise ValueError(f"ranking size must be positive, got k={k}")

    segment_match = _ANSWER_TAG_PAT.search(text or "")
    segment = segment_match.group(1) if segment_match else (text or "")
    array_texts = re.findall(r"\[[^\[\]]*\]", segment)
    if not array_texts:
        raise ValueError("no JSON ranking array found")

    valid_rankings: List[List[int]] = []
    invalid_reasons: List[str] = []
    expected = set(range(1, k + 1))
    for array_text in array_texts:
        try:
            value = json.loads(array_text)
        except json.JSONDecodeError as exc:
            invalid_reasons.append(f"invalid JSON array: {exc.msg}")
            continue
        if not isinstance(value, list):
            invalid_reasons.append("ranking is not a JSON list")
            continue
        if any(isinstance(x, bool) or not isinstance(x, int) for x in value):
            invalid_reasons.append("ranking contains non-integer indices")
            continue
        if len(value) != k:
            invalid_reasons.append(f"ranking length {len(value)} != expected {k}")
            continue
        if len(set(value)) != k:
            invalid_reasons.append("ranking contains duplicate indices")
            continue
        if set(value) != expected:
            invalid_reasons.append(f"ranking must be a permutation of 1..{k}")
            continue
        valid_rankings.append(value)

    if len(valid_rankings) == 1:
        return valid_rankings[0]
    if len(valid_rankings) > 1:
        raise ValueError("multiple valid ranking arrays found")
    reason = invalid_reasons[0] if invalid_reasons else "no valid ranking array found"
    raise ValueError(reason)
