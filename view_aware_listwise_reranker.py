"""
View-Aware Listwise Reranker Module

Provides a single callable rerank_view() that:
  - Takes k candidates sorted best-first by current score
  - Shuffles them into a random order before showing to the VLM
    (same motivation as listwise_rerank_shuffle.py: prevents position bias)
  - Calls the VLM via OpenAI-compatible API
  - Maps VLM output back through the shuffle permutation
  - Returns filenames in best-to-worst order (List[str])

Shuffle seed convention (reproducible, per-call unique):
    call_seed = shuffle_seed * 10**7 + int(query_id) * 10 + t
where t ∈ {1,2,3,4} is the iteration index.
Pass shuffle_seed=None to disable shuffling (candidates shown in score order).

Dependency: requires prompts.py in the same package or Python path.
"""

import os
import base64
import random
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


from prompts import get_system_prompt, format_user_prompt, parse_ranking_output, format_fashioniq_user_prompt


class RerankViewError(RuntimeError):
    """Persistent rerank failure with the complete request-attempt trace."""

    def __init__(
        self,
        message: str,
        *,
        display_order: List[str],
        attempts: List[Dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.display_order = list(display_order)
        self.attempts = list(attempts)


# ── Image helpers ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=60000)
def encode_image_cached(path: str, max_size: int = 0, quality: int = 85) -> str:
    """Base64-encode an image, optionally resizing to max_size on the long side.

    max_size=0 (default): read raw bytes, no re-encoding — identical to the
    previous behaviour used for CIRR/CIRCO.
    max_size>0: open with PIL, thumbnail to (max_size, max_size) preserving
    aspect ratio, then save as JPEG with the given quality. Both max_size and
    quality are part of the lru_cache key.
    """
    if max_size > 0:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _media_type(path: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(Path(path).suffix.lower(), "image/jpeg")


# ── Message builder ───────────────────────────────────────────────────────────

def build_view_messages(
    query: str,
    reference: Optional[str],
    candidates: List[str],   # absolute paths in display order (after shuffle)
    use_cot: bool = False,
    task_type: Optional[str] = None,
    image_max_size: int = 0,
    image_quality: int = 85,
) -> List[Dict[str, Any]]:
    """
    Build the OpenAI message list for one reranker call.

    Args:
        query:          modification / query text
        reference:      absolute path to reference image (None for text-only tasks)
        candidates:     absolute image paths in the order shown to the VLM;
                        Candidate i in the prompt = candidates[i-1].
        use_cot:        chain-of-thought flag
        task_type:      "cir" | "text2img" | "visdial" | None.
                        None → auto-detect: "cir" if reference present, else "text2img".
        image_max_size: long-side pixel cap before base64 (0 = no resize, default).
        image_quality:  JPEG re-encode quality when image_max_size > 0 (default 85).

    Returns:
        OpenAI-compatible messages list  [{role, content}, ...]
    """
    k = len(candidates)
    has_ref = reference is not None

    effective_task = task_type if task_type is not None else ("cir" if has_ref else "text2img")

    system_prompt = get_system_prompt(effective_task)
    if effective_task == "fashioniq":
        user_text = format_fashioniq_user_prompt(query, k)
    else:
        user_text = format_user_prompt(
            query_text=query,
            k=k,
            use_cot=use_cot,
            output_format="ranking",
            has_reference_image=has_ref,
            task_type=effective_task,
        )

    content: List[Dict] = []

    if has_ref:
        ref_b64 = encode_image_cached(reference, image_max_size, image_quality)
        content = [
            {"type": "text", "text": "Reference Image:"},
            {"type": "image_url", "image_url": {"url": f"data:{_media_type(reference)};base64,{ref_b64}"}},
            {"type": "text", "text": f"Modification: {query}"},
        ]
    else:
        content = [{"type": "text", "text": f"Query: {query}"}]

    content.append({"type": "text", "text": f"\nCandidates (K={k}):"})
    for i, path in enumerate(candidates, start=1):
        b64 = encode_image_cached(path, image_max_size, image_quality)
        content += [
            {"type": "text", "text": f"\nCandidate {i}:"},
            {"type": "image_url", "image_url": {"url": f"data:{_media_type(path)};base64,{b64}"}},
        ]
    content.append({"type": "text", "text": f"\n{user_text}"})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": content},
    ]


# ── Main callable ─────────────────────────────────────────────────────────────

def rerank_view(
    client: OpenAI,
    model_name: str,
    query: str,
    reference_filename: Optional[str],
    candidates: List[str],        # filenames sorted best-first by current score
    image_dir: str,
    shuffle_seed: Optional[int] = None,   # deterministic per-call seed; None = no shuffle
    use_cot: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.0,
    max_retries: int = 3,
    task_type: Optional[str] = None,
    image_max_size: int = 0,
    image_quality: int = 85,
    return_details: bool = False,
) -> Any:
    """
    Call the VLM listwise reranker on a single view with optional shuffle.

    Args:
        client:              OpenAI client pointing at the vLLM server
        model_name:          model identifier
        query:               modification / query text
        reference_filename:  reference image filename (relative), or None
        candidates:          k filenames in current-score-descending order
        image_dir:           directory containing all images
        shuffle_seed:        if not None, shuffle candidates before calling VLM
                             using random.Random(shuffle_seed) — fully reproducible
        use_cot:             chain-of-thought flag (default False)
        max_tokens:          max new tokens
        temperature:         sampling temperature (0 = greedy)
        max_retries:         retry count on API errors
        task_type:           "cir" | "text2img" | "visdial" | None (auto-detect).
        image_max_size:      resize long side to this many pixels before base64
                             encoding (0 = no resize, default — preserves existing
                             CIRR/CIRCO behaviour).
        image_quality:       JPEG quality when image_max_size > 0 (default 85).

    Returns:
        List[str] of length k: filenames ordered best-to-worst by VLM judgment.
        If return_details=True, also returns the actual prompt order and raw
        model text so callers can persist full per-call traces.
        Invalid ranking formats fall back to the pre-call candidate order and
        are marked parse_valid=False. API failures still follow max_retries.
    """
    k = len(candidates)
    ref_path = os.path.join(image_dir, reference_filename) if reference_filename else None

    # ── Shuffle ────────────────────────────────────────────────────────────────
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        display_order = list(candidates)
        rng.shuffle(display_order)
    else:
        display_order = list(candidates)

    # display_order[i-1] is shown as "Candidate i" to the VLM
    cand_paths = [os.path.join(image_dir, fn) for fn in display_order]

    attempts: List[Dict[str, Any]] = []
    try:
        msgs = build_view_messages(
            query=query,
            reference=ref_path,
            candidates=cand_paths,
            use_cot=use_cot,
            task_type=task_type,
            image_max_size=image_max_size,
            image_quality=image_quality,
        )
    except Exception as exc:
        raise RerankViewError(
            f"failed to build multimodal prompt: {exc}",
            display_order=display_order,
            attempts=attempts,
        ) from exc

    for attempt in range(max_retries):
        started_at_unix = time.time()
        started = time.perf_counter()
        attempt_record: Dict[str, Any] = {
            "attempt": attempt + 1,
            "started_at_unix": started_at_unix,
            "latency_seconds": None,
            "api_success": False,
            "parse_success": False,
        }
        stage = "api_call"
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"enable_thinking": False},
            )
            text = resp.choices[0].message.content or ""
            attempt_record["api_success"] = True
            attempt_record["raw_response"] = text
            usage = getattr(resp, "usage", None)
            if usage is not None:
                if hasattr(usage, "model_dump"):
                    usage_dict = usage.model_dump()
                elif hasattr(usage, "dict"):
                    usage_dict = usage.dict()
                else:
                    usage_dict = None
                if usage_dict:
                    attempt_record["usage"] = {
                        key: value for key, value in usage_dict.items()
                        if value is not None
                    }
            stage = "parse_response"
            # raw_ranking[j] = 1-based position in display_order of the (j+1)-th best
            try:
                raw_ranking = parse_ranking_output(text, k=k)
            except ValueError as parse_exc:
                attempt_record["latency_seconds"] = time.perf_counter() - started
                attempt_record["error_stage"] = stage
                attempt_record["error_type"] = type(parse_exc).__name__
                attempt_record["error_message"] = str(parse_exc)
                attempts.append(attempt_record)
                fallback_ranking = list(candidates)
                if return_details:
                    return {
                        "ranked_fns": fallback_ranking,
                        "display_order": display_order,
                        "raw_response": text,
                        "raw_ranking": None,
                        "attempts": attempts,
                        "parse_valid": False,
                        "fallback_used": True,
                        "fallback_type": "pre_call_order",
                        "fallback_reason": str(parse_exc),
                    }
                return fallback_ranking
            attempt_record["parse_success"] = True
            attempt_record["parsed_ranking"] = list(raw_ranking)
            # Map back to filenames: position p → display_order[p-1]
            ranked_fns = [display_order[pos - 1] for pos in raw_ranking]
            attempt_record["latency_seconds"] = time.perf_counter() - started
            attempts.append(attempt_record)
            if return_details:
                return {
                    "ranked_fns": ranked_fns,
                    "display_order": display_order,
                    "raw_response": text,
                    "raw_ranking": raw_ranking,
                    "attempts": attempts,
                    "parse_valid": True,
                    "fallback_used": False,
                }
            return ranked_fns
        except Exception as exc:
            attempt_record["latency_seconds"] = time.perf_counter() - started
            attempt_record["error_stage"] = stage
            attempt_record["error_type"] = type(exc).__name__
            attempt_record["error_message"] = str(exc)
            attempts.append(attempt_record)
            if attempt == max_retries - 1:
                raise RerankViewError(
                    f"VLM API failed after {max_retries} attempts: {exc}",
                    display_order=display_order,
                    attempts=attempts,
                ) from exc
