"""AMBER: HOM candidate views, adaptive query scheduling, and Elo ranking.

The original experiment core and current scheduler are combined in this file.
Use --mode scheduler-only for the paper main method.
"""

import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from view_aware_listwise_reranker import RerankViewError, rerank_view


# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_API_URL          = "http://localhost:8005/v1"
DEFAULT_API_KEY          = "EMPTY"
DEFAULT_MODEL            = "qwen-vl"
DEFAULT_VIEW_SIZE        = 10
DEFAULT_TOP_N            = 50
DEFAULT_MAX_TOKENS       = 512
DEFAULT_TEMPERATURE      = 0
DEFAULT_MAX_WORKERS      = 32
DEFAULT_MAX_RETRIES      = 3
DEFAULT_SHUFFLE_SEED     = 42
DEFAULT_BUDGET_PER_QUERY = 5
DEFAULT_BATCH_SIZE       = 100
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 1800.0

DEFAULT_ELO_BASE_RATING  = 1500.0
DEFAULT_ELO_RATING_RANGE = 400.0
DEFAULT_ELO_K            = 32.0
DEFAULT_ELO_SCALE        = 400.0
DEFAULT_ELO_MAX_PAIR_GAP = 0

# candidate-level sampling value
DEFAULT_CAND_SV          = "head_obs"
DEFAULT_CAND_MARGIN_C    = 50.0
DEFAULT_CAND_MARGIN_K    = 2
DEFAULT_CAND_TOPK        = 10

# query-level uncertainty
DEFAULT_QUERY_SV         = "delta_bar"
DEFAULT_MARGIN_C         = 50.0
DEFAULT_MARGIN_ALPHA     = 0.7
DEFAULT_MARGIN_K2        = 5
DEFAULT_ENTROPY_L        = 10
DEFAULT_ENTROPY_TAU      = 50.0
DEFAULT_VIEW_HEAD_ALPHA  = 0.5
DEFAULT_HEAD_DISP_D      = 10
DEFAULT_HEAD_DISP_STOP_THRESHOLD = 0.0
DEFAULT_RBO_D            = 20
DEFAULT_RBO_P            = 0.9
DEFAULT_RBO_STOP_THRESHOLD = 0.0
DEFAULT_BUDGET_STOP_MAX_PER_QUERY_MULTIPLIER = 5.0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_input(path: str) -> Tuple[Dict, List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("config", {}), data.get("cases", [])


def get_pool(case: Dict, top_n: int) -> List[str]:
    """返回按嵌入排名顺序的候选列表（长度 <= top_n）。"""
    cand_key = next(
        (k for k in ["retrieved_top50", "retrieved_top100", "retrieved_top20", "candidates"]
         if k in case),
        None,
    )
    if cand_key is None:
        return []
    return list(case[cand_key][:top_n])


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


# ── Elo Initialization ────────────────────────────────────────────────────────

def init_elo_scores(
    pool: List[str],
    base_rating: float = DEFAULT_ELO_BASE_RATING,
    rating_range: float = DEFAULT_ELO_RATING_RANGE,
) -> Dict[str, float]:
    """线性初始化 Elo 分数：embedding rank 第 0 名得 base+range，末名得 base。"""
    n = len(pool)
    if n <= 1:
        return {fn: base_rating for fn in pool}
    return {
        fn: base_rating + rating_range * (n - 1 - i) / (n - 1)
        for i, fn in enumerate(pool)
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _query_int(query_id: str) -> int:
    try:
        return int(query_id)
    except ValueError:
        return abs(hash(query_id)) % 10 ** 6


# ── Per-query state ───────────────────────────────────────────────────────────

@dataclass
class QueryState:
    """单 query 的持久化状态，跨多轮 rerank 复用。"""

    query_id:   str
    query:      str
    reference:  Optional[str]
    target:     Optional[str]

    pool:       List[str]
    img_to_idx: Dict[str, int]

    m_stats:    Dict[str, int]
    sum_delta:  Dict[str, float]
    delta_bar:  Dict[str, float]
    observed:   Set[str]

    global_ranks: Dict[str, int]

    qid_int:        int
    round:          int = 0
    rerank_rounds:  List[Dict] = field(default_factory=list)
    failed_rounds:  List[Dict] = field(default_factory=list)

    last_view_delta_bar: Optional[float] = None
    last_head_rank_displacement: Optional[float] = None
    last_rbo_displacement: Optional[float] = None

    elo_scores: Dict[str, float] = field(default_factory=dict)

    error: Optional[str] = None


def init_query_state(
    case: Dict,
    top_n: int,
    elo_base_rating: float = DEFAULT_ELO_BASE_RATING,
    elo_rating_range: float = DEFAULT_ELO_RATING_RANGE,
) -> QueryState:
    query_id  = str(case.get("id") or case.get("query_id", ""))
    query     = case.get("query", "")
    reference = case.get("reference")
    target    = case.get("target")

    pool = get_pool(case, top_n)
    img_to_idx = {fn: i for i, fn in enumerate(pool)}

    m_stats:   Dict[str, int]   = {fn: 0   for fn in pool}
    sum_delta: Dict[str, float] = {fn: 0.0 for fn in pool}
    delta_bar: Dict[str, float] = {fn: 0.0 for fn in pool}

    elo_scores   = init_elo_scores(pool, elo_base_rating, elo_rating_range)
    # initial global_ranks = embedding order
    global_ranks = {fn: i + 1 for i, fn in enumerate(pool)}

    return QueryState(
        query_id=query_id,
        query=query,
        reference=reference,
        target=target,
        pool=pool,
        img_to_idx=img_to_idx,
        m_stats=m_stats,
        sum_delta=sum_delta,
        delta_bar=delta_bar,
        observed=set(),
        global_ranks=global_ranks,
        elo_scores=elo_scores,
        qid_int=_query_int(query_id),
        error=None if pool else "empty candidate pool",
    )


# ── Elo update ────────────────────────────────────────────────────────────────

def _update_elo_from_ranking(
    state: QueryState,
    ranked_fns: List[str],
    K: float = DEFAULT_ELO_K,
    elo_scale: float = DEFAULT_ELO_SCALE,
    max_pair_gap: int = DEFAULT_ELO_MAX_PAIR_GAP,
) -> None:
    """根据 VLM 输出的局部排序对 state.elo_scores 做在线 Elo 更新。

    对每个 pairwise win (i ≻ j)：
        E_i = 1 / (1 + 10^((s_j - s_i) / elo_scale))
        s_i ← s_i + K * (1 - E_i)
        s_j ← s_j - K * (1 - E_i)

    max_pair_gap > 0 时只展开 rank 距离 <= max_pair_gap 的 pair。
    """
    n = len(ranked_fns)
    for rank_a in range(n):
        fn_a = ranked_fns[rank_a]
        if fn_a not in state.elo_scores:
            continue
        for rank_b in range(rank_a + 1, n):
            if max_pair_gap > 0 and rank_b - rank_a > max_pair_gap:
                break
            fn_b = ranked_fns[rank_b]
            if fn_b not in state.elo_scores:
                continue
            s_a = state.elo_scores[fn_a]
            s_b = state.elo_scores[fn_b]
            e_a = 1.0 / (1.0 + 10.0 ** ((s_b - s_a) / elo_scale))
            delta = K * (1.0 - e_a)
            state.elo_scores[fn_a] = s_a + delta
            state.elo_scores[fn_b] = s_b - delta


# ── Global ranks from Elo ─────────────────────────────────────────────────────

def _elo_global_ranks(state: QueryState) -> Dict[str, int]:
    """Elo 分数排序 observed，unobserved 按 embedding 顺序追加。"""
    observed = [fn for fn in state.pool if fn in state.observed]
    observed_sorted = sorted(
        observed,
        key=lambda fn: (-state.elo_scores.get(fn, DEFAULT_ELO_BASE_RATING),
                        state.img_to_idx[fn]),
    )
    unobserved = [fn for fn in state.pool if fn not in state.observed]
    full_order = observed_sorted + unobserved
    return {fn: rank + 1 for rank, fn in enumerate(full_order)}


# ── Candidate-level Sampling Value ───────────────────────────────────────────

def _head_obs_margin_factors(
    fn: str,
    state: QueryState,
    cand_margin_c: float = DEFAULT_CAND_MARGIN_C,
    cand_margin_k: int   = DEFAULT_CAND_MARGIN_K,
) -> Tuple[float, float, float]:
    """Return (head, obs, margin) factors used by head_obs_margin."""
    r_i = state.global_ranks[fn]
    m_i = state.m_stats[fn]
    head_factor = 1.0 / math.log2(r_i + 1)
    obs_factor = 1.0 / math.sqrt(m_i + 1)

    sorted_pool = sorted(state.pool, key=lambda x: state.global_ranks[x])
    pos = r_i - 1  # 0-indexed position in sorted list
    lo = max(0, pos - cand_margin_k)
    hi = min(len(sorted_pool) - 1, pos + cand_margin_k)

    elo_i = state.elo_scores.get(fn, DEFAULT_ELO_BASE_RATING)
    min_gap = math.inf
    for p in range(lo, hi + 1):
        if p == pos:
            continue
        neighbor = sorted_pool[p]
        gap = abs(elo_i - state.elo_scores.get(neighbor, DEFAULT_ELO_BASE_RATING))
        if gap < min_gap:
            min_gap = gap

    if math.isinf(min_gap):
        min_gap = 0.0
    margin_factor = 1.0 / (1.0 + min_gap / max(cand_margin_c, 1e-9))
    return head_factor, obs_factor, margin_factor


def _cand_sv_head_obs(
    fn: str,
    state: QueryState,
    **_kwargs: Any,
) -> float:
    r_i = state.global_ranks[fn]
    m_i = state.m_stats[fn]
    return (1.0 / math.log2(r_i + 1)) * (1.0 / math.sqrt(m_i + 1))


def _cand_sv_head_obs_margin(
    fn: str,
    state: QueryState,
    cand_margin_c: float = DEFAULT_CAND_MARGIN_C,
    cand_margin_k: int   = DEFAULT_CAND_MARGIN_K,
    **_kwargs: Any,
) -> float:
    """head_obs × Elo margin factor.

    g_i = min_{j in N_K(i)} |elo_i - elo_j|
    factor = 1 / (1 + g_i / c)
    """
    head_factor, obs_factor, margin_factor = _head_obs_margin_factors(
        fn, state, cand_margin_c, cand_margin_k
    )
    return head_factor * obs_factor * margin_factor


def _cand_sv_hom_drop_head(
    fn: str,
    state: QueryState,
    cand_margin_c: float = DEFAULT_CAND_MARGIN_C,
    cand_margin_k: int   = DEFAULT_CAND_MARGIN_K,
    **_kwargs: Any,
) -> float:
    """Ablation: remove the head-rank factor from head_obs_margin."""
    _head_factor, obs_factor, margin_factor = _head_obs_margin_factors(
        fn, state, cand_margin_c, cand_margin_k
    )
    return obs_factor * margin_factor


def _cand_sv_hom_drop_obs(
    fn: str,
    state: QueryState,
    cand_margin_c: float = DEFAULT_CAND_MARGIN_C,
    cand_margin_k: int   = DEFAULT_CAND_MARGIN_K,
    **_kwargs: Any,
) -> float:
    """Ablation: remove the observation-count factor from head_obs_margin."""
    head_factor, _obs_factor, margin_factor = _head_obs_margin_factors(
        fn, state, cand_margin_c, cand_margin_k
    )
    return head_factor * margin_factor


def _cand_sv_hom_drop_margin(
    fn: str,
    state: QueryState,
    cand_margin_c: float = DEFAULT_CAND_MARGIN_C,
    cand_margin_k: int   = DEFAULT_CAND_MARGIN_K,
    **_kwargs: Any,
) -> float:
    """Ablation: remove the Elo-margin factor from head_obs_margin."""
    head_factor, obs_factor, _margin_factor = _head_obs_margin_factors(
        fn, state, cand_margin_c, cand_margin_k
    )
    return head_factor * obs_factor


def _cand_sv_top10(
    fn: str,
    state: QueryState,
    **_kwargs: Any,
) -> float:
    """Fixed top-10 ablation: score only by current global rank."""
    rank = state.global_ranks[fn]
    if rank > DEFAULT_CAND_TOPK:
        return -math.inf
    return float(DEFAULT_CAND_TOPK + 1 - rank)


CAND_SV_REGISTRY: Dict[str, Callable[..., float]] = {
    "head_obs":        _cand_sv_head_obs,
    "head_obs_margin": _cand_sv_head_obs_margin,
    "hom_drop_head":   _cand_sv_hom_drop_head,
    "hom_drop_obs":    _cand_sv_hom_drop_obs,
    "hom_drop_margin": _cand_sv_hom_drop_margin,
    "top10":           _cand_sv_top10,
}


def build_view_by_sampling_value(
    pool: List[str],
    state: QueryState,
    view_size: int,
    cand_sv: str = DEFAULT_CAND_SV,
    cand_sv_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    cand_sv_kwargs = cand_sv_kwargs or {}
    sv_fn = CAND_SV_REGISTRY.get(cand_sv, _cand_sv_head_obs)
    ranked = sorted(
        pool,
        key=lambda fn: sv_fn(fn, state, **cand_sv_kwargs),
        reverse=True,
    )
    if cand_sv == "top10":
        ranked = [
            fn for fn in ranked
            if state.global_ranks[fn] <= DEFAULT_CAND_TOPK
        ]
    return ranked[:view_size]


def order_view_candidates(
    state: QueryState,
    view: List[str],
) -> List[str]:
    """按 Elo 分数对 view 内候选排序（observed 先，unobserved 按 embedding 顺序）。"""
    observed_sorted = sorted(
        [fn for fn in view if fn in state.observed],
        key=lambda fn: (-state.elo_scores.get(fn, DEFAULT_ELO_BASE_RATING),
                        state.img_to_idx[fn]),
    )
    unobserved = [fn for fn in view if fn not in state.observed]
    return observed_sorted + unobserved


# ── Query-level Uncertainty ───────────────────────────────────────────────────

def _query_sv_delta_bar(
    state: QueryState,
    **_kwargs: Any,
) -> float:
    """最近一轮 view 内归一化平均位次变化。"""
    if state.error is not None:
        return -math.inf
    if state.last_view_delta_bar is None:
        return 1.0
    return float(max(0.0, min(1.0, state.last_view_delta_bar)))


def _query_sv_uniform_budget(
    state: QueryState,
    **_kwargs: Any,
) -> float:
    """Uniform-budget ablation: keep scores tied so scheduling balances rounds."""
    if state.error is not None:
        return -math.inf
    return 1.0


def _head_weighted_rank_displacement(
    old_ranks: Dict[str, int],
    new_ranks: Dict[str, int],
    D: int = DEFAULT_HEAD_DISP_D,
) -> float:
    """Head-weighted displacement over the union of previous/current top-D."""
    D = max(1, D)
    old_top_d = {fn for fn, rank in old_ranks.items() if rank <= D}
    new_top_d = {fn for fn, rank in new_ranks.items() if rank <= D}
    top_union = old_top_d | new_top_d
    if not top_union:
        return 0.0

    numerator = 0.0
    denominator = 0.0
    for fn in top_union:
        old_rank = old_ranks[fn]
        new_rank = new_ranks[fn]
        weight = 1.0 / math.log2(1.0 + min(old_rank, new_rank))
        numerator += weight * (abs(new_rank - old_rank) / D)
        denominator += weight

    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _rbo_truncated(
    old_ranks: Dict[str, int],
    new_ranks: Dict[str, int],
    D: int = DEFAULT_RBO_D,
    p: float = DEFAULT_RBO_P,
) -> float:
    """截断 RBO（Rank-Biased Overlap），计算两个排名列表前 D 名的相似度，归一化到 [0, 1]。

    原始公式：
        RBO_raw = (1-p) * sum_{d=1}^{D} p^{d-1} * |top-d(old) ∩ top-d(new)| / d

    当两列表完全相同时 RBO_raw = 1 - p^D（而非 1），故对最大值归一化：
        RBO_norm = RBO_raw / (1 - p^D)

    值域严格 [0, 1]，值越高表示越相似（排名越稳定）。
    p 越大越重视深层排名，p 越小越聚焦头部（推荐 p=0.9）。
    """
    D = max(1, D)
    p = max(0.0, min(1.0 - 1e-9, p))

    old_list = sorted(old_ranks, key=lambda fn: old_ranks[fn])[:D]
    new_list = sorted(new_ranks, key=lambda fn: new_ranks[fn])[:D]

    rbo = 0.0
    p_d = 1.0          # p^(d-1)
    old_set: set = set()
    new_set: set = set()
    for d in range(1, D + 1):
        if d <= len(old_list):
            old_set.add(old_list[d - 1])
        if d <= len(new_list):
            new_set.add(new_list[d - 1])
        rbo += p_d * (len(old_set & new_set) / d)
        p_d *= p

    rbo_raw = (1.0 - p) * rbo
    # 归一化：当两列表相同时 rbo_raw = 1 - p^D，除以最大值得到 [0, 1]
    rbo_max = 1.0 - p ** D
    return rbo_raw / rbo_max if rbo_max > 0.0 else 0.0


def _rbo_displacement(
    old_ranks: Dict[str, int],
    new_ranks: Dict[str, int],
    D: int = DEFAULT_RBO_D,
    p: float = DEFAULT_RBO_P,
) -> float:
    """1 - RBO，值越大表示排名变化越大（不确定性越高）。"""
    return 1.0 - _rbo_truncated(old_ranks, new_ranks, D=D, p=p)


def _query_sv_head_rank_displacement(
    state: QueryState,
    **_kwargs: Any,
) -> float:
    """最近一轮全局 top-D 并集上的头部加权 rank displacement。"""
    if state.error is not None:
        return -math.inf
    if state.last_head_rank_displacement is None:
        return 1.0
    return float(max(0.0, state.last_head_rank_displacement))


def _query_sv_head_rank_displacement_budget_stop(
    state: QueryState,
    **kwargs: Any,
) -> float:
    """head_rank_displacement 的预算停止版，分数本身保持一致。"""
    return _query_sv_head_rank_displacement(state, **kwargs)


def _query_sv_rbo(
    state: QueryState,
    **_kwargs: Any,
) -> float:
    """1 - RBO（截断到前 D 名的 Rank-Biased Overlap）。值越大表示排名越不稳定。"""
    if state.error is not None:
        return -math.inf
    if state.last_rbo_displacement is None:
        return 1.0
    return float(max(0.0, state.last_rbo_displacement))


def _query_sv_rbo_budget_stop(
    state: QueryState,
    **kwargs: Any,
) -> float:
    """rbo 的预算停止版，分数本身保持一致。"""
    return _query_sv_rbo(state, **kwargs)


def _query_sv_topk_margin(
    state: QueryState,
    margin_c: float     = DEFAULT_MARGIN_C,
    margin_alpha: float = DEFAULT_MARGIN_ALPHA,
    margin_k2: int      = DEFAULT_MARGIN_K2,
    **_kwargs: Any,
) -> float:
    """U_q = alpha * exp(-m_1/c) + (1-alpha) * exp(-m_K/c).

    m_k = elo[rank_k] - elo[rank_{k+1}]  (Elo gap at rank boundary k/k+1)
    Uses all candidates (observed and unobserved) ordered by current global_ranks.
    """
    if state.error is not None:
        return -math.inf

    sorted_pool = sorted(state.pool, key=lambda fn: state.global_ranks[fn])
    elo_list = [state.elo_scores.get(fn, DEFAULT_ELO_BASE_RATING)
                for fn in sorted_pool]

    def _gap(k: int) -> float:
        """Gap between rank-k and rank-(k+1) candidates (1-indexed)."""
        idx = k - 1
        if idx < 0 or idx + 1 >= len(elo_list):
            return 0.0
        return max(0.0, elo_list[idx] - elo_list[idx + 1])

    m1 = _gap(1)
    mk = _gap(margin_k2)
    c  = max(margin_c, 1e-9)
    return margin_alpha * math.exp(-m1 / c) + (1.0 - margin_alpha) * math.exp(-mk / c)


def _query_sv_head_entropy(
    state: QueryState,
    entropy_l:   int   = DEFAULT_ENTROPY_L,
    entropy_tau: float = DEFAULT_ENTROPY_TAU,
    **_kwargs: Any,
) -> float:
    """归一化 softmax 熵：H_q / log(L)，值域 [0, 1]。

    p_i = exp((elo_i - elo_1) / tau) / Z   for i=1..L
    H_q = -sum p_i log p_i
    U_q = H_q / log(L)
    """
    if state.error is not None:
        return -math.inf

    sorted_pool = sorted(state.pool, key=lambda fn: state.global_ranks[fn])
    L = min(entropy_l, len(sorted_pool))
    if L < 2:
        return 0.0

    top_pool = sorted_pool[:L]
    elo_vals = np.array(
        [state.elo_scores.get(fn, DEFAULT_ELO_BASE_RATING) for fn in top_pool],
        dtype=np.float64,
    )
    tau = max(entropy_tau, 1e-9)
    shifted = (elo_vals - elo_vals[0]) / tau  # subtract max for numerical stability
    exp_vals = np.exp(shifted)
    probs = exp_vals / exp_vals.sum()

    # clip to avoid log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    H = float(-np.sum(probs * np.log(probs)))
    return H / math.log(L)


def _binary_entropy(p: float) -> float:
    """归一化二元熵，p=0.5 时为 1，p 越接近 0/1 越低。"""
    p = max(1e-12, min(1.0 - 1e-12, p))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)


def _next_selected_view(
    state: QueryState,
    selected_view: Optional[List[str]] = None,
    view_size: int = DEFAULT_VIEW_SIZE,
    cand_sv: str = DEFAULT_CAND_SV,
    cand_sv_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if selected_view is not None:
        return selected_view
    view = build_view_by_sampling_value(
        state.pool,
        state,
        view_size,
        cand_sv,
        cand_sv_kwargs,
    )
    return order_view_candidates(state, view)


def _query_sv_view_pair_entropy(
    state: QueryState,
    selected_view: Optional[List[str]] = None,
    view_size: int = DEFAULT_VIEW_SIZE,
    cand_sv: str = DEFAULT_CAND_SV,
    cand_sv_kwargs: Optional[Dict[str, Any]] = None,
    elo_scale: float = DEFAULT_ELO_SCALE,
    view_head_alpha: float = DEFAULT_VIEW_HEAD_ALPHA,
    **_kwargs: Any,
) -> float:
    """下一轮 selected view 的 pairwise Elo 胜率熵，并偏向全局头部候选。

    pairwise 部分衡量本轮实际会比较的 candidate 两两胜负不确定性；
    head 部分用 1/log2(rank+1) 的均值给包含头部候选的 view 更高权重。
    """
    if state.error is not None:
        return -math.inf

    view = _next_selected_view(
        state,
        selected_view=selected_view,
        view_size=view_size,
        cand_sv=cand_sv,
        cand_sv_kwargs=cand_sv_kwargs,
    )
    if len(view) < 2:
        return 0.0

    scale = max(elo_scale, 1e-9)
    entropies: List[float] = []
    for i, fn_i in enumerate(view):
        s_i = state.elo_scores.get(fn_i, DEFAULT_ELO_BASE_RATING)
        for fn_j in view[i + 1:]:
            s_j = state.elo_scores.get(fn_j, DEFAULT_ELO_BASE_RATING)
            p_i = 1.0 / (1.0 + 10.0 ** ((s_j - s_i) / scale))
            entropies.append(_binary_entropy(p_i))

    pair_entropy = float(np.mean(entropies)) if entropies else 0.0
    head_score = float(np.mean([
        1.0 / math.log2(state.global_ranks[fn] + 1)
        for fn in view
    ]))
    #alpha = max(0.0, min(1.0, view_head_alpha))
    head_factor = math.sqrt(head_score)
    return pair_entropy * head_factor


QUERY_SV_REGISTRY: Dict[str, Callable[..., float]] = {
    "delta_bar":                  _query_sv_delta_bar,
    "head_rank_displacement":     _query_sv_head_rank_displacement,
    "head_rank_displacement_budget_stop": _query_sv_head_rank_displacement_budget_stop,
    "rbo":                        _query_sv_rbo,
    "rbo_budget_stop":            _query_sv_rbo_budget_stop,
    "uniform_budget":             _query_sv_uniform_budget,
    "topk_margin":                _query_sv_topk_margin,
    "head_entropy":               _query_sv_head_entropy,
    "view_pair_entropy":          _query_sv_view_pair_entropy,
}


def compute_uncertainty(
    state: QueryState,
    query_sv: str = DEFAULT_QUERY_SV,
    query_sv_kwargs: Optional[Dict[str, Any]] = None,
    selected_view: Optional[List[str]] = None,
) -> float:
    query_sv_kwargs = query_sv_kwargs or {}
    sv_fn = QUERY_SV_REGISTRY.get(query_sv, _query_sv_delta_bar)
    return sv_fn(state, selected_view=selected_view, **query_sv_kwargs)


# ── Aggregation (Elo only) ────────────────────────────────────────────────────

def aggregate_elo(state: QueryState) -> List[str]:
    """按 elo_scores 排序 observed 候选（高分优先，tiebreak 用 embedding rank）。"""
    observed = [fn for fn in state.pool if fn in state.observed]
    if not observed:
        return []
    return sorted(
        observed,
        key=lambda fn: (
            -state.elo_scores.get(fn, DEFAULT_ELO_BASE_RATING),
            state.img_to_idx[fn],
        ),
    )


# ── Round executor ────────────────────────────────────────────────────────────

def run_one_round(
    state: QueryState,
    client: OpenAI,
    model_name: str,
    image_dir: str,
    view_size: int,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    shuffle_seed: int,
    elo_k: float = DEFAULT_ELO_K,
    elo_scale: float = DEFAULT_ELO_SCALE,
    elo_max_pair_gap: int = DEFAULT_ELO_MAX_PAIR_GAP,
    head_disp_d: int = DEFAULT_HEAD_DISP_D,
    rbo_d: int = DEFAULT_RBO_D,
    rbo_p: float = DEFAULT_RBO_P,
    cand_sv: str = DEFAULT_CAND_SV,
    cand_sv_kwargs: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None,
    image_max_size: int = 0,
    image_quality: int = 85,
) -> None:
    """对一个 query 跑 1 轮（= 1 次 rerank_view 调用），原地更新 state。"""
    if state.error is not None:
        return

    round_started = time.perf_counter()
    t = state.round + 1
    call_seed = shuffle_seed * 10 ** 7 + state.qid_int * 10 + t
    prev_global_ranks = dict(state.global_ranks)

    view = build_view_by_sampling_value(
        state.pool, state, view_size, cand_sv, cand_sv_kwargs
    )
    sorted_view = order_view_candidates(state, view)
    r_old_map = {fn: j + 1 for j, fn in enumerate(sorted_view)}
    pre_rerank_seconds = time.perf_counter() - round_started

    try:
        rerank_result = rerank_view(
            client=client,
            model_name=model_name,
            query=state.query,
            reference_filename=state.reference,
            candidates=sorted_view,
            image_dir=image_dir,
            shuffle_seed=call_seed,
            use_cot=False,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            task_type=task_type,
            image_max_size=image_max_size,
            image_quality=image_quality,
            return_details=True,
        )
        ranked_fns = rerank_result["ranked_fns"]
    except Exception as exc:
        display_order = list(getattr(exc, "display_order", sorted_view))
        attempts = list(getattr(exc, "attempts", []))
        state.failed_rounds.append({
            "t": t,
            "selected_candidates": list(view),
            "input_ranking": list(sorted_view),
            "prompt_input_order": display_order,
            "shuffle_seed": call_seed,
            "attempts": attempts,
            "timing": {
                "pre_rerank_seconds": pre_rerank_seconds,
                "post_rerank_seconds": 0.0,
                "total_round_seconds": time.perf_counter() - round_started,
            },
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        state.error = f"rerank_view failed at round {t}: {exc}"
        return

    state.round = t
    post_rerank_started = time.perf_counter()

    if not rerank_result.get("parse_valid", True):
        state.last_head_rank_displacement = 0.0
        state.last_rbo_displacement = 0.0
        state.last_view_delta_bar = 0.0
        state.rerank_rounds.append({
            "t": t,
            "status": "fallback_pre_call",
            "parse_valid": False,
            "fallback_used": True,
            "fallback_type": rerank_result.get("fallback_type", "pre_call_order"),
            "fallback_reason": rerank_result.get("fallback_reason"),
            "selected_candidates": list(view),
            "input_ranking": list(sorted_view),
            "prompt_input_order": list(rerank_result["display_order"]),
            "model_output_ranking": list(sorted_view),
            "shuffle_seed": call_seed,
            "attempts": list(rerank_result.get("attempts", [])),
            "timing": {
                "pre_rerank_seconds": pre_rerank_seconds,
                "post_rerank_seconds": time.perf_counter() - post_rerank_started,
                "total_round_seconds": time.perf_counter() - round_started,
            },
            "hrd": 0.0,
            "rbo": 0.0,
            "vdb": 0.0,
        })
        return

    r_new_map = {fn: rank + 1 for rank, fn in enumerate(ranked_fns)}

    for fn in ranked_fns:
        state.observed.add(fn)

    _update_elo_from_ranking(
        state, ranked_fns,
        K=elo_k,
        elo_scale=elo_scale,
        max_pair_gap=elo_max_pair_gap,
    )

    view_delta_ts: List[float] = []
    for fn in sorted_view:
        delta_t = abs(r_old_map[fn] - r_new_map[fn]) / max(len(sorted_view) - 1, 1)
        state.m_stats[fn]   += 1
        state.sum_delta[fn] += delta_t
        state.delta_bar[fn]  = state.sum_delta[fn] / state.m_stats[fn]
        view_delta_ts.append(delta_t)

    state.last_view_delta_bar = (
        float(np.mean(view_delta_ts)) if view_delta_ts else None
    )

    state.global_ranks = _elo_global_ranks(state)
    state.last_head_rank_displacement = _head_weighted_rank_displacement(
        prev_global_ranks,
        state.global_ranks,
        D=head_disp_d,
    )
    state.last_rbo_displacement = _rbo_displacement(
        prev_global_ranks,
        state.global_ranks,
        D=rbo_d,
        p=rbo_p,
    )

    state.rerank_rounds.append({
        "t": t,
        "status": "success",
        "parse_valid": True,
        "fallback_used": False,
        "selected_candidates": list(view),
        "input_ranking": list(sorted_view),
        "prompt_input_order": list(rerank_result["display_order"]),
        "model_output_ranking": list(ranked_fns),
        "shuffle_seed": call_seed,
        "attempts": list(rerank_result.get("attempts", [])),
        "timing": {
            "pre_rerank_seconds": pre_rerank_seconds,
            "post_rerank_seconds": time.perf_counter() - post_rerank_started,
            "total_round_seconds": time.perf_counter() - round_started,
        },
        "hrd": (
            round(state.last_head_rank_displacement, 6)
            if state.last_head_rank_displacement is not None else None
        ),
        "rbo": (
            round(state.last_rbo_displacement, 6)
            if state.last_rbo_displacement is not None else None
        ),
        "vdb": (
            round(state.last_view_delta_bar, 6)
            if state.last_view_delta_bar is not None else None
        ),
    })


# ── Finalize ──────────────────────────────────────────────────────────────────

def finalize_query(state: QueryState) -> Dict[str, Any]:
    """构造最终 ranking：observed 按 Elo 排序，unobserved 按 embedding 顺序追加。"""
    observed_sorted = aggregate_elo(state)
    unobserved = [fn for fn in state.pool if fn not in state.observed]
    result: Dict[str, Any] = {
        "query_id": state.query_id,
        "final_ranking": observed_sorted + unobserved,
        "actual_rounds": state.round,
        "valid_rerank_rounds": sum(
            item.get("parse_valid", True) for item in state.rerank_rounds
        ),
        "fallback_rerank_rounds": sum(
            not item.get("parse_valid", True) for item in state.rerank_rounds
        ),
        "rerank_rounds": state.rerank_rounds,
    }
    if state.failed_rounds:
        result["failed_rounds"] = state.failed_rounds
    if state.error is not None:
        result["error"] = state.error
    return result


class CheckpointSaver:
    """Periodically writes current query states to an atomic JSON checkpoint."""

    def __init__(
        self,
        path: Optional[str],
        interval_seconds: float,
        states: List[QueryState],
        metadata: Dict[str, Any],
    ) -> None:
        self.path = Path(path) if path else None
        self.interval_seconds = interval_seconds
        self.states = states
        self.metadata = metadata
        self.lock = threading.Lock()
        self.next_save_at = time.monotonic() + max(interval_seconds, 0.0)

    def maybe_save(self, force: bool = False) -> None:
        if self.path is None:
            return
        if self.interval_seconds <= 0.0 and not force:
            return

        now = time.monotonic()
        if not force and now < self.next_save_at:
            return

        with self.lock:
            now = time.monotonic()
            if not force and now < self.next_save_at:
                return
            self._write()
            self.next_save_at = now + max(self.interval_seconds, 0.0)

    def _write(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "saved_at_unix": time.time(),
            "metadata": self.metadata,
            "summary": {
                "num_queries": len(self.states),
                "finished_queries": sum(
                    1 for s in self.states if s.error is not None or s.round > 0
                ),
                "total_rounds": sum(s.round for s in self.states),
                "successful_rerank_rounds": sum(
                    len(s.rerank_rounds) for s in self.states
                ),
                "failed_rounds": sum(len(s.failed_rounds) for s in self.states),
                "api_attempts": sum(
                    len(round_item.get("attempts", []))
                    for state in self.states
                    for round_item in state.rerank_rounds + state.failed_rounds
                ),
            },
            "results": [finalize_query(s) for s in self.states],
        }
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False)
        os.replace(tmp_path, self.path)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_phase1(
    states: List[QueryState],
    round_kwargs: Dict[str, Any],
    query_sv: str,
    query_sv_kwargs: Dict[str, Any],
    max_workers: int,
    checkpoint_saver: Optional[CheckpointSaver] = None,
) -> None:
    """Phase 1: 并行跑每个 query 1 轮。"""
    def _work(s: QueryState) -> None:
        run_one_round(s, **round_kwargs)

    valid_states = [s for s in states if s.error is None or s.error == "empty candidate pool"]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        pbar = tqdm(total=len(valid_states), desc="Phase 1 (all queries ×1)")
        for _ in ex.map(_work, valid_states):
            pbar.update(1)
            if checkpoint_saver is not None:
                checkpoint_saver.maybe_save()
        pbar.close()

def run_phase2(
    states: List[QueryState],
    total_budget: int,
    spent: int,
    batch_size: int,
    max_per_query: int,
    stop_uncertainty_threshold: float,
    round_kwargs: Dict[str, Any],
    query_sv: str,
    query_sv_kwargs: Dict[str, Any],
    max_workers: int,
    checkpoint_saver: Optional[CheckpointSaver] = None,
    schedule_log: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, List[Dict]]:
    """Phase 2: 批量优先队列，按 query_sv 不确定性降序，严格控制 spent <= total_budget。"""
    global_log: List[Dict] = schedule_log if schedule_log is not None else []
    iter_idx = 0
    pbar = tqdm(total=total_budget - spent, desc=f"Phase 2 [{query_sv}]")

    while spent < total_budget:
        iter_idx += 1
        scheduler_started = time.perf_counter()
        cand = [
            s for s in states
            if s.error is None
            and (max_per_query == 0 or s.round < max_per_query)
        ]
        if not cand:
            break

        scored: List[Tuple[float, QueryState]] = [
            (compute_uncertainty(s, query_sv, query_sv_kwargs), s) for s in cand
        ]

        # per-query 早停：每个 query 不确定性低于阈值后永久移出候选池
        if stop_uncertainty_threshold > 0.0:
            scored = [(u, s) for u, s in scored if u >= stop_uncertainty_threshold]
            if not scored:
                global_log.append({
                    "iter":       iter_idx,
                    "spent":      spent,
                    "stop_reason": "all_queries_converged",
                    "threshold":  stop_uncertainty_threshold,
                })
                break

        scored.sort(key=lambda x: (-x[0], x[1].round, x[1].qid_int))

        k = min(batch_size, total_budget - spent, len(scored))
        if k <= 0:
            break
        batch = scored[:k]
        scheduler_seconds = time.perf_counter() - scheduler_started
        spent_before = spent
        selected_before = [
            {
                "query_id": s.query_id,
                "uncertainty": float(u),
                "round_before": s.round,
            }
            for u, s in batch
        ]

        def _work(s: QueryState) -> None:
            run_one_round(s, **round_kwargs)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for _ in ex.map(_work, [s for _, s in batch]):
                pass

        spent += k
        pbar.update(k)
        global_log.append({
            "iter": iter_idx,
            "k": k,
            "spent_before": spent_before,
            "spent_after": spent,
            "u_max": float(batch[0][0]),
            "u_min": float(batch[-1][0]),
            "scheduler_seconds": scheduler_seconds,
            "selected": [
                {
                    **before,
                    "round_after": state.round,
                    "success": state.error is None,
                    "error": state.error,
                }
                for before, (_uncertainty, state) in zip(selected_before, batch)
            ],
        })
        if checkpoint_saver is not None:
            checkpoint_saver.maybe_save()

    pbar.close()
    return spent, global_log


def run_uniform_budget(
    states: List[QueryState],
    budget_per_query: int,
    total_budget: int,
    use_total_budget: bool,
    max_per_query: int,
    round_kwargs: Dict[str, Any],
    query_sv: str,
    query_sv_kwargs: Dict[str, Any],
    max_workers: int,
    checkpoint_saver: Optional[CheckpointSaver] = None,
) -> Tuple[int, List[Dict]]:
    """Uniform-budget mode: each query runs serial rounds; queries run in parallel."""
    valid_states = [s for s in states if s.error is None]
    if not valid_states:
        return 0, []

    targets: Dict[str, int] = {}
    if use_total_budget:
        base = total_budget // len(valid_states)
        remainder = total_budget % len(valid_states)
        for idx, s in enumerate(valid_states):
            targets[s.query_id] = base + (1 if idx < remainder else 0)
    else:
        for s in valid_states:
            targets[s.query_id] = budget_per_query

    if max_per_query > 0:
        targets = {qid: min(target, max_per_query) for qid, target in targets.items()}

    planned_calls = sum(targets.values())
    if planned_calls <= 0:
        return 0, []

    def _work(s: QueryState) -> int:
        target_rounds = targets[s.query_id]
        rounds_before = s.round
        while s.error is None and s.round < target_rounds:
            run_one_round(s, **round_kwargs)
            if checkpoint_saver is not None:
                checkpoint_saver.maybe_save()
        return max(0, s.round - rounds_before)

    pbar = tqdm(total=planned_calls, desc="Uniform budget (per-query serial)")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for done_rounds in ex.map(_work, valid_states):
            pbar.update(done_rounds)
    pbar.close()

    spent = sum(s.round for s in valid_states)
    global_log = [{
        "mode": "uniform_budget",
        "planned_calls": planned_calls,
        "spent": spent,
        "target_min": min(targets.values()),
        "target_max": max(targets.values()),
    }]
    return spent, global_log


# ── Main ──────────────────────────────────────────────────────────────────────

def run_core() -> None:
    parser = argparse.ArgumentParser(
        description="View-Aware Reranking v15 – Pure Elo + Pluggable Sampling Value",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Candidate-level sampling value (--cand_sv):
  head_obs         1/log2(r+1) * 1/sqrt(m+1)                          [default]
  head_obs_margin  以上 * 1/(1 + g_i/c), g_i = Elo 邻域最小 rating 差距
  hom_drop_head    head_obs_margin 去掉 1/log2(r+1) 头部排名因子
  hom_drop_obs     head_obs_margin 去掉 1/sqrt(m+1) 观测次数因子
  hom_drop_margin  head_obs_margin 去掉 1/(1 + g_i/c) Elo margin 因子
  top10            只 rerank 当前全局排序前 10 名

Query-level uncertainty (--query_sv):
  delta_bar        最近一轮 view 内归一化平均位次变化                    [default]
  head_rank_displacement
                   最近一轮全局 top-D 并集上的头部加权 rank displacement
  head_rank_displacement_budget_stop
                   head_rank_displacement 的预算停止版，支持阈值早停；
                   默认单 query 最大轮数为 5 × 平均 budget
  uniform_budget   平均分配 budget：所有 query 同分，轮数少者优先
  topk_margin      alpha*exp(-m_1/c) + (1-alpha)*exp(-m_K/c)
  head_entropy     前 L 名 Elo 分数的 softmax 熵（归一化到 [0,1]）
  view_pair_entropy
                   下一轮 selected view 内 pairwise 胜率熵均值 × 头部排名权重
""",
    )

    parser.add_argument("--input",  type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="运行中 checkpoint JSON 路径；默认使用 <output>.checkpoint.json")
    parser.add_argument("--checkpoint_interval_seconds", type=float,
                        default=DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
                        help="checkpoint 保存间隔秒数；<=0 表示不定时保存（默认 300）")
    parser.add_argument("--schedule_log_path", type=str, default=None,
                        help="完整全局 query 调度日志；默认使用 <output>.schedule.json")

    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--model",   type=str, default=DEFAULT_MODEL)

    parser.add_argument("--view_size",   type=int, default=DEFAULT_VIEW_SIZE)
    parser.add_argument("--top_n",       type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--num_queries", type=int, default=None,
                        help="限制处理前 N 条 query（调试用）")

    parser.add_argument("--total_budget",     type=int, default=0,
                        help="全局总预算（VLM 调用次数）。>0 时覆盖 --budget_per_query")
    parser.add_argument("--budget_per_query", type=int, default=DEFAULT_BUDGET_PER_QUERY,
                        help="平均每 query 预算，total = this × num_queries（默认 5）")
    parser.add_argument("--batch_size",       type=int, default=DEFAULT_BATCH_SIZE,
                        help="Phase 2 每次挑多少 top-K 不确定 query（默认 100）")
    parser.add_argument("--max_per_query",    type=int, default=0,
                        help=(
                            "单 query 最大轮数上限，0 表示不限制；"
                            "head_rank_displacement_budget_stop 下默认自动设为 "
                            "budget_stop_max_per_query_multiplier × 平均 budget"
                        ))

    parser.add_argument("--max_tokens",   type=int,   default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature",  type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_workers",  type=int,   default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max_retries",  type=int,   default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--shuffle_seed",    type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--image_dir",       type=str, default=None)
    parser.add_argument("--image_max_size",  type=int, default=0,
                        help="缩图长边上限（px），0=不缩（默认）")
    parser.add_argument("--image_quality",   type=int, default=85,
                        help="JPEG 重编码质量，仅 image_max_size>0 时生效（默认 85）")

    # ── Elo 超参数 ────────────────────────────────────────────────────────────
    parser.add_argument("--elo_base_rating",  type=float, default=DEFAULT_ELO_BASE_RATING,
                        help="Elo 初始分基准值（默认 1500.0）")
    parser.add_argument("--elo_rating_range", type=float, default=DEFAULT_ELO_RATING_RANGE,
                        help="Elo 初始分线性范围：最佳=base+range，最差=base（默认 400.0）")
    parser.add_argument("--elo_k",            type=float, default=DEFAULT_ELO_K,
                        help="Elo 更新步长 K（默认 32.0）")
    parser.add_argument("--elo_scale",        type=float, default=DEFAULT_ELO_SCALE,
                        help="Elo 期望胜率公式尺度参数（默认 400.0）")
    parser.add_argument("--elo_max_pair_gap", type=int,   default=DEFAULT_ELO_MAX_PAIR_GAP,
                        help="展开 pairwise 时最大 rank 距离，0=全部展开（默认 0）")

    # ── Candidate-level 采样价值 ──────────────────────────────────────────────
    parser.add_argument("--cand_sv", type=str, default=DEFAULT_CAND_SV,
                        choices=list(CAND_SV_REGISTRY.keys()),
                        help="candidate-level 采样价值策略（默认 head_obs）")
    parser.add_argument("--cand_margin_c", type=float, default=DEFAULT_CAND_MARGIN_C,
                        help="head_obs_margin 的 Elo 差距尺度 c（默认 50.0）")
    parser.add_argument("--cand_margin_k", type=int,   default=DEFAULT_CAND_MARGIN_K,
                        help="head_obs_margin 的邻域大小 K，上下各 K 名（默认 2）")

    # ── Query-level 不确定性 ──────────────────────────────────────────────────
    parser.add_argument("--query_sv", type=str, default=DEFAULT_QUERY_SV,
                        choices=list(QUERY_SV_REGISTRY.keys()),
                        help="query-level 不确定性策略（默认 delta_bar）")

    # topk_margin 参数
    parser.add_argument("--head_disp_d", type=int, default=DEFAULT_HEAD_DISP_D,
                        help="head_rank_displacement 使用的 top-D 深度（默认 10）")
    parser.add_argument("--head_disp_stop_threshold", type=float,
                        default=DEFAULT_HEAD_DISP_STOP_THRESHOLD,
                        help=(
                            "head_rank_displacement_budget_stop 的早停阈值；"
                            "0 表示不启用（默认 0）"
                        ))
    parser.add_argument("--rbo_d", type=int, default=DEFAULT_RBO_D,
                        help="rbo/rbo_budget_stop 截断深度 D（默认 20）")
    parser.add_argument("--rbo_p", type=float, default=DEFAULT_RBO_P,
                        help="rbo 持久性参数 p，越大越重视深层排名（默认 0.9）")
    parser.add_argument("--rbo_stop_threshold", type=float,
                        default=DEFAULT_RBO_STOP_THRESHOLD,
                        help=(
                            "rbo_budget_stop 的早停阈值（1-RBO）；"
                            "0 表示不启用（默认 0）"
                        ))
    parser.add_argument("--budget_stop_max_per_query_multiplier", type=float,
                        default=DEFAULT_BUDGET_STOP_MAX_PER_QUERY_MULTIPLIER,
                        help=(
                            "head_rank_displacement_budget_stop 在 --max_per_query=0 时，"
                            "自动设置单 query 最大轮数为该倍数 × 平均 budget（默认 5.0）"
                        ))
    parser.add_argument("--margin_c",     type=float, default=DEFAULT_MARGIN_C,
                        help="topk_margin 的 Elo 差距尺度 c（默认 50.0）")
    parser.add_argument("--margin_alpha", type=float, default=DEFAULT_MARGIN_ALPHA,
                        help="topk_margin 中 R@1 边界的权重 alpha（默认 0.7）")
    parser.add_argument("--margin_k2",    type=int,   default=DEFAULT_MARGIN_K2,
                        help="topk_margin 中第二个边界的 K（默认 5，即 R@5）")

    # head_entropy 参数
    parser.add_argument("--entropy_l",   type=int,   default=DEFAULT_ENTROPY_L,
                        help="head_entropy 使用的前 L 名（默认 10）")
    parser.add_argument("--entropy_tau", type=float, default=DEFAULT_ENTROPY_TAU,
                        help="head_entropy softmax 温度 tau（默认 50.0）")
    parser.add_argument("--view_head_alpha", type=float, default=DEFAULT_VIEW_HEAD_ALPHA,
                        help="view_pair_entropy 中头部排名权重强度，0=只用 pairwise 熵，1=完全乘头部均值权重（默认 0.5）")

    args = parser.parse_args()
    run_started_at_unix = time.time()
    run_started = time.perf_counter()

    # ── 加载数据 ───────────────────────────────────────────────────────────────
    print(f"Loading: {args.input}")
    config, cases = load_input(args.input)

    image_dir = args.image_dir or config.get("image_dir", "")
    if not image_dir or not os.path.exists(image_dir):
        raise ValueError(
            f"image_dir not found: {image_dir!r}. 请通过 --image_dir 指定图片目录。"
        )

    _dataset = config.get("dataset", "").lower()
    if "fashioniq" in _dataset:
        task_type = "fashioniq"
    elif config.get("is_cir_task", True):
        task_type = "cir"
    else:
        task_type = "text2img"

    if args.num_queries:
        cases = cases[: args.num_queries]

    num_queries  = len(cases)
    total_budget = (
        args.total_budget if args.total_budget > 0
        else args.budget_per_query * num_queries
    )
    if total_budget < num_queries:
        raise ValueError(
            f"total_budget={total_budget} 小于 num_queries={num_queries}，"
            "Phase 1 至少需要每 query 1 次调用。"
        )
    avg_budget = total_budget / max(num_queries, 1)
    effective_max_per_query = args.max_per_query
    _budget_stop_modes = {"head_rank_displacement_budget_stop", "rbo_budget_stop"}
    if args.query_sv in _budget_stop_modes and effective_max_per_query == 0:
        effective_max_per_query = max(
            1,
            int(math.ceil(args.budget_stop_max_per_query_multiplier * avg_budget)),
        )

    cand_sv_kwargs: Dict[str, Any] = {
        "cand_margin_c": args.cand_margin_c,
        "cand_margin_k": args.cand_margin_k,
    }
    query_sv_kwargs: Dict[str, Any] = {
        "margin_c":     args.margin_c,
        "margin_alpha": args.margin_alpha,
        "margin_k2":    args.margin_k2,
        "entropy_l":    args.entropy_l,
        "entropy_tau":  args.entropy_tau,
        "head_disp_d":  args.head_disp_d,
        "view_size":    args.view_size,
        "cand_sv":      args.cand_sv,
        "cand_sv_kwargs": cand_sv_kwargs,
        "elo_scale":    args.elo_scale,
        "view_head_alpha": args.view_head_alpha,
        "rbo_d":        args.rbo_d,
        "rbo_p":        args.rbo_p,
    }

    print("Configuration:")
    print(f"  Dataset             : {config.get('dataset', 'unknown')}")
    print(f"  Image dir           : {image_dir}")
    print(f"  Queries             : {num_queries}")
    print(f"  View size M         : {args.view_size}")
    print(f"  Candidate pool N    : {args.top_n}")
    print(f"  Elo K / scale       : {args.elo_k} / {args.elo_scale}")
    print(f"  Elo base / range    : {args.elo_base_rating} / {args.elo_rating_range}")
    print(f"  Elo max_pair_gap    : "
          f"{'all' if args.elo_max_pair_gap == 0 else args.elo_max_pair_gap}")
    print(f"  Cand sampling value : {args.cand_sv}", end="")
    if args.cand_sv in {
        "head_obs_margin",
        "hom_drop_head",
        "hom_drop_obs",
        "hom_drop_margin",
    }:
        print(f"  (c={args.cand_margin_c}, k={args.cand_margin_k})", end="")
    elif args.cand_sv == "top10":
        print(f"  (fixed top-{DEFAULT_CAND_TOPK})", end="")
    print()
    print(f"  Query uncertainty   : {args.query_sv}", end="")
    if args.query_sv == "topk_margin":
        print(f"  (c={args.margin_c}, alpha={args.margin_alpha}, k2={args.margin_k2})", end="")
    elif args.query_sv in {"head_rank_displacement", "head_rank_displacement_budget_stop"}:
        print(f"  (D={args.head_disp_d})", end="")
        if args.query_sv == "head_rank_displacement_budget_stop":
            print(
                f", stop_threshold={args.head_disp_stop_threshold}, "
                f"max_per_query={effective_max_per_query}",
                end="",
            )
    elif args.query_sv in {"rbo", "rbo_budget_stop"}:
        print(f"  (D={args.rbo_d}, p={args.rbo_p})", end="")
        if args.query_sv == "rbo_budget_stop":
            print(
                f", stop_threshold={args.rbo_stop_threshold}, "
                f"max_per_query={effective_max_per_query}",
                end="",
            )
    elif args.query_sv == "head_entropy":
        print(f"  (L={args.entropy_l}, tau={args.entropy_tau})", end="")
    elif args.query_sv == "view_pair_entropy":
        print(f"  (head_alpha={args.view_head_alpha})", end="")
    print()
    print(f"  Total budget        : {total_budget} "
          f"(= {avg_budget:.2f} per query avg)")
    print(f"  Batch size          : {args.batch_size}")
    print(f"  Max per-query cap   : "
          f"{'unlimited' if effective_max_per_query == 0 else effective_max_per_query}")
    print(f"  VLM                 : {args.model} @ {args.api_url}")
    print(f"  Max workers         : {args.max_workers}")
    print(f"  Shuffle seed        : {args.shuffle_seed}")
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None and args.checkpoint_interval_seconds > 0.0:
        checkpoint_path = f"{args.output}.checkpoint.json"
    print(f"  Checkpoint          : "
          f"{checkpoint_path or 'disabled'}"
          f" (interval={args.checkpoint_interval_seconds}s)")
    schedule_log_path = args.schedule_log_path or f"{args.output}.schedule.json"
    print(f"  Schedule log        : {schedule_log_path}")
    print()

    client = OpenAI(api_key=args.api_key, base_url=args.api_url)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 初始化 state ──────────────────────────────────────────────────────────
    print("Initializing query states ...")
    states: List[QueryState] = [
        init_query_state(
            c, args.top_n,
            elo_base_rating=args.elo_base_rating,
            elo_rating_range=args.elo_rating_range,
        )
        for c in cases
    ]
    empty_cnt = sum(1 for s in states if s.error == "empty candidate pool")
    if empty_cnt:
        print(f"  Warning: {empty_cnt} queries have empty candidate pool; they will be skipped.")

    phase1_query_ids = [s.query_id for s in states if s.error is None]
    schedule_trace: Dict[str, Any] = {
        "schema_version": 1,
        "dataset": config.get("dataset", "unknown"),
        "input": args.input,
        "output": args.output,
        "query_policy": args.query_sv,
        "candidate_policy": args.cand_sv,
        "batch_size": args.batch_size,
        "budget_per_query": args.budget_per_query,
        "total_budget": total_budget,
        "max_per_query": effective_max_per_query,
        "seed": args.shuffle_seed,
        "phase1": {
            "selected_query_ids": phase1_query_ids,
            "selected": len(phase1_query_ids),
            "accepted": None,
            "failed": None,
        },
        "phase2": [],
        "summary": None,
    }

    checkpoint_saver = CheckpointSaver(
        path=checkpoint_path,
        interval_seconds=args.checkpoint_interval_seconds,
        states=states,
        metadata={
            "input": args.input,
            "output": args.output,
            "dataset": config.get("dataset", "unknown"),
            "model": args.model,
            "api_url": args.api_url,
            "query_sv": args.query_sv,
            "cand_sv": args.cand_sv,
            "budget_per_query": args.budget_per_query,
            "total_budget": total_budget,
            "max_per_query": effective_max_per_query,
            "view_size": args.view_size,
            "top_n": args.top_n,
            "batch_size": args.batch_size,
            "elo_base_rating": args.elo_base_rating,
            "elo_rating_range": args.elo_rating_range,
            "elo_k": args.elo_k,
            "elo_scale": args.elo_scale,
            "elo_max_pair_gap": args.elo_max_pair_gap,
            "shuffle_seed": args.shuffle_seed,
            "image_dir": image_dir,
            "image_max_size": args.image_max_size,
            "image_quality": args.image_quality,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "schedule_trace": schedule_trace,
        },
    )

    round_kwargs: Dict[str, Any] = dict(
        client=client,
        model_name=args.model,
        image_dir=image_dir,
        view_size=args.view_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        shuffle_seed=args.shuffle_seed,
        elo_k=args.elo_k,
        elo_scale=args.elo_scale,
        elo_max_pair_gap=args.elo_max_pair_gap,
        head_disp_d=args.head_disp_d,
        rbo_d=args.rbo_d,
        rbo_p=args.rbo_p,
        cand_sv=args.cand_sv,
        cand_sv_kwargs=cand_sv_kwargs,
        task_type=task_type,
        image_max_size=args.image_max_size,
        image_quality=args.image_quality,
    )

    global_log: List[Dict] = []
    if args.query_sv == "uniform_budget":
        spent, global_log = run_uniform_budget(
            states=states,
            budget_per_query=args.budget_per_query,
            total_budget=total_budget,
            use_total_budget=args.total_budget > 0,
            max_per_query=effective_max_per_query,
            round_kwargs=round_kwargs,
            query_sv=args.query_sv,
            query_sv_kwargs=query_sv_kwargs,
            max_workers=args.max_workers,
            checkpoint_saver=checkpoint_saver,
        )
        print(f"Uniform budget done. Spent = {spent} / {total_budget} "
              f"(global iters = {len(global_log)})")
    else:
        # ── Phase 1 ───────────────────────────────────────────────────────────
        run_phase1(
            states,
            round_kwargs,
            args.query_sv,
            query_sv_kwargs,
            args.max_workers,
            checkpoint_saver=checkpoint_saver,
        )
        spent = sum(1 for s in states if s.round >= 1)
        phase1_failed = sum(1 for s in states if s.error is not None)
        schedule_trace["phase1"]["accepted"] = spent
        schedule_trace["phase1"]["failed"] = phase1_failed
        print(f"Phase 1 done. Spent = {spent} / {total_budget}")

        # ── Phase 2 ───────────────────────────────────────────────────────────
        if spent < total_budget:
            spent, global_log = run_phase2(
                states=states,
                total_budget=total_budget,
                spent=spent,
                batch_size=args.batch_size,
                max_per_query=effective_max_per_query,
                stop_uncertainty_threshold=(
                    args.head_disp_stop_threshold
                    if args.query_sv == "head_rank_displacement_budget_stop"
                    else args.rbo_stop_threshold
                    if args.query_sv == "rbo_budget_stop"
                    else 0.0
                ),
                round_kwargs=round_kwargs,
                query_sv=args.query_sv,
                query_sv_kwargs=query_sv_kwargs,
                max_workers=args.max_workers,
                checkpoint_saver=checkpoint_saver,
                schedule_log=schedule_trace["phase2"],
            )
        print(f"Phase 2 done. Spent = {spent} / {total_budget} "
              f"(global iters = {len(global_log)})")

    # ── Finalize & summary ────────────────────────────────────────────────────
    results = [finalize_query(s) for s in states]

    error_cnt = sum(1 for r in results if r.get("error"))
    valid     = [r for r in results if not r.get("error")]
    rounds    = [r.get("actual_rounds", 0) for r in valid]
    accepted_rounds = sum(r.get("actual_rounds", 0) for r in results)
    valid_rerank_rounds = sum(r.get("valid_rerank_rounds", 0) for r in results)
    fallback_rerank_rounds = sum(r.get("fallback_rerank_rounds", 0) for r in results)
    api_attempts = sum(
        len(round_item.get("attempts", []))
        for result in results
        for round_item in result.get("rerank_rounds", []) + result.get("failed_rounds", [])
    )
    run_wall_seconds = time.perf_counter() - run_started

    if rounds:
        r_min    = int(np.min(rounds))
        r_max    = int(np.max(rounds))
        r_mean   = float(np.mean(rounds))
        r_median = float(np.median(rounds))
        r_std    = float(np.std(rounds))
    else:
        r_min = r_max = 0
        r_mean = r_median = r_std = 0.0

    bucket1 = sum(1 for r in rounds if r == 1)
    bucket2 = sum(1 for r in rounds if r == 2)
    bucket3 = sum(1 for r in rounds if 3 <= r <= 5)
    bucket4 = sum(1 for r in rounds if r >= 6)

    print()
    print(f"Processed {len(results)} queries | Errors: {error_cnt}")
    print(f"Total VLM calls        : {spent}  (budget={total_budget})")
    print(f"Accepted rerank rounds : {accepted_rounds}")
    print(f"Valid ranking rounds   : {valid_rerank_rounds}")
    print(f"Pre-call fallbacks     : {fallback_rerank_rounds}")
    print(f"Underlying API attempts: {api_attempts}")
    print(f"Run wall time (seconds): {run_wall_seconds:.3f}")
    print(f"Rounds per query       : "
          f"min={r_min}, median={r_median:.1f}, mean={r_mean:.2f}, "
          f"max={r_max}, std={r_std:.2f}")
    print(f"Round distribution     : "
          f"1-round={bucket1}, 2-rounds={bucket2}, "
          f"3-5-rounds={bucket3}, 6+-rounds={bucket4}")

    if valid:
        sample = valid[0]
        print(f"Sample (query_id={sample['query_id']}): "
              f"actual_rounds={sample.get('actual_rounds','?')}, "
              f"top-5={sample['final_ranking'][:5]}")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    schedule_trace["summary"] = {
        "queries": len(results),
        "selected_logical_rounds": spent,
        "accepted_rounds": accepted_rounds,
        "valid_rerank_rounds": valid_rerank_rounds,
        "fallback_rerank_rounds": fallback_rerank_rounds,
        "api_attempts": api_attempts,
        "errors": error_cnt,
        "phase2_iterations": len(schedule_trace["phase2"]),
        "run_started_at_unix": run_started_at_unix,
        "run_ended_at_unix": time.time(),
        "wall_seconds": run_wall_seconds,
        "scheduler_seconds": sum(
            float(item.get("scheduler_seconds") or 0.0)
            for item in schedule_trace["phase2"]
        ),
        "api_attempt_seconds": sum(
            float(attempt.get("latency_seconds") or 0.0)
            for result in results
            for round_item in result.get("rerank_rounds", []) + result.get("failed_rounds", [])
            for attempt in round_item.get("attempts", [])
        ),
        "non_api_round_seconds": sum(
            max(
                0.0,
                float(round_item.get("timing", {}).get("total_round_seconds") or 0.0)
                - sum(
                    float(attempt.get("latency_seconds") or 0.0)
                    for attempt in round_item.get("attempts", [])
                ),
            )
            for result in results
            for round_item in result.get("rerank_rounds", []) + result.get("failed_rounds", [])
        ),
        "round_min": r_min,
        "round_median": r_median,
        "round_mean": r_mean,
        "round_max": r_max,
    }
    write_json_atomic(Path(schedule_log_path), schedule_trace)

    checkpoint_saver.maybe_save(force=True)

    print(f"\nSaved {len(results)} results → {args.output}")
    if checkpoint_saver.path is not None:
        print(f"Checkpoint saved → {checkpoint_saver.path}")
    print(f"Schedule log saved → {schedule_log_path}")
    print()
    print("Use make_submission.py for CIRCO/CIRR conversion.")

    if error_cnt or accepted_rounds != total_budget or spent != total_budget:
        raise SystemExit(
            "Run artifacts were saved, but the run is invalid: "
            f"errors={error_cnt}, selected={spent}, accepted={accepted_rounds}, "
            f"expected={total_budget}."
        )



# Current AMBER query scheduler. The original core routines above are reused.
import hashlib
import re
import shlex
import sys

amber = sys.modules[__name__]

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "config/protocol.json"
PAIR_QUERY_POLICY = "pair_marginal_view"
PAIR_CANDIDATE_POLICY = "pair_greedy"
ORIGINAL_BUILD_VIEW = amber.build_view_by_sampling_value
ORIGINAL_RUN_ONE_ROUND = amber.run_one_round
ORIGINAL_FINALIZE_QUERY = amber.finalize_query


def _evidence(state: amber.QueryState) -> dict[tuple[int, int], float]:
    if not hasattr(state, "pair_evidence"):
        state.pair_evidence = {}
    return state.pair_evidence


def _pair_key(state: amber.QueryState, first: str, second: str) -> tuple[int, int]:
    i, j = state.img_to_idx[first], state.img_to_idx[second]
    return (i, j) if i < j else (j, i)


def pair_uncertainty(first_rating: float, second_rating: float, elo_scale: float) -> float:
    """Return 4p(1-p) for the existing base-10 Elo win probability."""
    if elo_scale <= 0:
        raise ValueError("elo_scale must be positive")
    log_odds = (first_rating - second_rating) * math.log(10.0) / elo_scale
    # p(1-p) is symmetric; this form avoids overflow for large rating gaps.
    tail = math.exp(-abs(log_odds))
    return 4.0 * tail / (1.0 + tail) ** 2


def pair_marginal_utility(
    state: amber.QueryState,
    first: str,
    second: str,
    elo_scale: float,
) -> float:
    """w_ij * log(1 + z_ij / (1 + C_ij)), with natural logarithm."""
    if first == second:
        return 0.0
    head_first = 1.0 / math.log2(state.global_ranks[first] + 1)
    head_second = 1.0 / math.log2(state.global_ranks[second] + 1)
    weight = (head_first + head_second) / 2.0
    z = pair_uncertainty(state.elo_scores[first], state.elo_scores[second], elo_scale)
    evidence = _evidence(state).get(_pair_key(state, first, second), 0.0)
    return weight * math.log1p(z / (1.0 + evidence))


def view_utility(state: amber.QueryState, view: list[str], elo_scale: float) -> float:
    return sum(
        pair_marginal_utility(state, first, second, elo_scale)
        for position, first in enumerate(view)
        for second in view[position + 1 :]
    )


def greedy_view(state: amber.QueryState, view_size: int, elo_scale: float) -> list[str]:
    """Seed with the best pair, then add the largest marginal-gain candidate.

    All ties follow the original embedding order, so the choice is reproducible.
    """
    pool = state.pool
    n = len(pool)
    target = min(max(view_size, 0), n)
    if target == 0:
        return []
    if target == 1 or n == 1:
        return pool[:1]

    utility = [[0.0] * n for _ in range(n)]
    best_pair = (0, 1)
    best_value = -math.inf
    for i in range(n):
        for j in range(i + 1, n):
            value = pair_marginal_utility(state, pool[i], pool[j], elo_scale)
            utility[i][j] = utility[j][i] = value
            if value > best_value:
                best_value, best_pair = value, (i, j)

    chosen = list(best_pair)
    chosen_set = set(chosen)
    while len(chosen) < target:
        best_index = -1
        best_gain = -math.inf
        for candidate in range(n):
            if candidate in chosen_set:
                continue
            gain = sum(utility[candidate][member] for member in chosen)
            if gain > best_gain:
                best_gain, best_index = gain, candidate
        chosen.append(best_index)
        chosen_set.add(best_index)
    return [pool[index] for index in chosen]


def next_view_and_utility(
    state: amber.QueryState,
    view_size: int,
    candidate_policy: str,
    candidate_kwargs: dict[str, Any] | None,
    elo_scale: float,
) -> tuple[list[str], float]:
    """Cache an unchanged query's next view across scheduler batches."""
    cache_key = (state.round, view_size, candidate_policy, elo_scale)
    cached = getattr(state, "_www_next_view", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1], cached[2]

    if candidate_policy == PAIR_CANDIDATE_POLICY:
        view = greedy_view(state, view_size, elo_scale)
    else:
        view = ORIGINAL_BUILD_VIEW(
            state.pool, state, view_size, candidate_policy, candidate_kwargs
        )
    score = view_utility(state, view, elo_scale)
    state._www_next_view = (cache_key, view, score)
    return view, score


def _candidate_view(
    pool: list[str],
    state: amber.QueryState,
    view_size: int,
    cand_sv: str = amber.DEFAULT_CAND_SV,
    cand_sv_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    if cand_sv != PAIR_CANDIDATE_POLICY:
        return ORIGINAL_BUILD_VIEW(pool, state, view_size, cand_sv, cand_sv_kwargs)
    if pool is not state.pool and pool != state.pool:
        raise ValueError("pair_greedy requires the query's full candidate pool")
    scale = _candidate_view.elo_scale
    return list(next_view_and_utility(state, view_size, cand_sv, cand_sv_kwargs, scale)[0])


_candidate_view.elo_scale = amber.DEFAULT_ELO_SCALE


def _query_pair_marginal_view(
    state: amber.QueryState,
    selected_view: list[str] | None = None,
    view_size: int = amber.DEFAULT_VIEW_SIZE,
    cand_sv: str = amber.DEFAULT_CAND_SV,
    cand_sv_kwargs: dict[str, Any] | None = None,
    elo_scale: float = amber.DEFAULT_ELO_SCALE,
    **_kwargs: Any,
) -> float:
    if state.error is not None:
        return -math.inf
    if selected_view is not None:
        return view_utility(state, selected_view, elo_scale)
    return next_view_and_utility(
        state, view_size, cand_sv, cand_sv_kwargs, elo_scale
    )[1]


def _run_one_round_with_evidence(state: amber.QueryState, **kwargs: Any) -> None:
    if state.error is not None:
        return ORIGINAL_RUN_ONE_ROUND(state, **kwargs)

    view_size = kwargs["view_size"]
    candidate_policy = kwargs["cand_sv"]
    candidate_kwargs = kwargs.get("cand_sv_kwargs")
    elo_scale = kwargs["elo_scale"]
    view, utility_before = next_view_and_utility(
        state, view_size, candidate_policy, candidate_kwargs, elo_scale
    )
    increments = [
        (
            _pair_key(state, first, second),
            pair_uncertainty(
                state.elo_scores[first], state.elo_scores[second], elo_scale
            ),
        )
        for position, first in enumerate(view)
        for second in view[position + 1 :]
    ]
    previous_round = state.round
    try:
        ORIGINAL_RUN_ONE_ROUND(state, **kwargs)

        # A parsed ranking or pre-call fallback is an accepted VLM call.
        # Transport failures have no returned comparison and add no evidence.
        if state.round > previous_round:
            record = state.rerank_rounds[-1]
            if record["selected_candidates"] != view:
                raise RuntimeError("selected view changed between scheduling and VLM call")
            evidence = _evidence(state)
            for pair, z_before in increments:
                evidence[pair] = evidence.get(pair, 0.0) + z_before
            record["pair_utility_before"] = utility_before
            record["pair_evidence_added"] = sum(z for _, z in increments)
    finally:
        state._www_next_view = None


def _finalize_with_evidence(state: amber.QueryState) -> dict[str, Any]:
    result = ORIGINAL_FINALIZE_QUERY(state)
    result["pair_evidence"] = [
        [i, j, value]
        for (i, j), value in sorted(_evidence(state).items())
    ]
    return result


def install_algorithm_hooks(elo_scale: float) -> None:
    _candidate_view.elo_scale = elo_scale
    # This registry entry enables the existing argument parser; view selection
    # itself is performed by _candidate_view because it is set-valued.
    amber.CAND_SV_REGISTRY[PAIR_CANDIDATE_POLICY] = lambda *_args, **_kwargs: 0.0
    amber.QUERY_SV_REGISTRY[PAIR_QUERY_POLICY] = _query_pair_marginal_view
    amber.build_view_by_sampling_value = _candidate_view
    amber.run_one_round = _run_one_round_with_evidence
    amber.finalize_query = _finalize_with_evidence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    common = protocol["common"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scheduler-only",), default="scheduler-only")
    parser.add_argument("--dataset", choices=sorted(protocol["datasets"]), required=True)
    parser.add_argument("--budget", type=int, choices=common["budget_points"], required=True)
    parser.add_argument("--seed", type=int, default=common["primary_seed"])
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--retrieval", type=Path, default=None,
                        help="Override dataset retrieval path (data files are supplied separately)")
    parser.add_argument("--api_url", default=os.environ.get("AMBER_API_URL", common["api_url"]))
    parser.add_argument("--api_key", default=os.environ.get("AMBER_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("AMBER_MODEL", common["vlm_model"]))
    parser.add_argument("--image_dir", default=None,
                        help="Override the protocol image directory (relative to project root or absolute)")
    parser.add_argument("--max_workers", type=int, default=common["max_workers"])
    parser.add_argument("--max_tokens", type=int, default=common["max_tokens"],
                        help="Override output-token cap for a model-specific prompt protocol")
    parser.add_argument("--print_command", action="store_true")
    args = parser.parse_args()

    if args.num_queries is not None and args.num_queries <= 0:
        parser.error("--num_queries must be positive")
    if args.max_workers <= 0:
        parser.error("--max_workers must be positive")
    if args.max_tokens <= 0:
        parser.error("--max_tokens must be positive")
    if args.run_tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_tag):
        parser.error("--run_tag may contain only ASCII letters, digits, dot, underscore, and hyphen")

    dataset = protocol["datasets"][args.dataset]
    retrieval = args.retrieval or (ROOT / dataset["retrieval"])
    if not retrieval.is_absolute():
        retrieval = ROOT / retrieval
    retrieval = retrieval.resolve()
    if not retrieval.is_file():
        parser.error(f"retrieval input not found: {retrieval}")
    image_dir = Path(args.image_dir or dataset["image_dir"])
    if not image_dir.is_absolute():
        image_dir = ROOT / image_dir
    image_dir = image_dir.resolve()
    max_per_query = min(common["top_n"], 2 * args.budget)
    seed_dir = f"seed{args.seed}" + (f"_{args.run_tag}" if args.run_tag else "")
    default_output = (
        ROOT / "outputs/www_revision" / args.mode / args.dataset
        / f"b{args.budget}" / seed_dir
        / f"amber_{args.dataset}_b{args.budget}_seed{args.seed}.jsonl"
    )
    output = (args.output or default_output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        parser.error(f"output must stay under {ROOT}")
    checkpoint = Path(f"{output}.checkpoint.json")
    schedule_log = Path(f"{output}.schedule.json")
    manifest_path = Path(f"{output}.run_manifest.json")
    collisions = [p for p in (output, checkpoint, schedule_log, manifest_path) if p.exists()]
    if collisions and not args.print_command:
        parser.error("refusing to overwrite: " + ", ".join(str(p) for p in collisions))

    candidate_policy = (
        PAIR_CANDIDATE_POLICY if args.mode == "unified" else common["candidate_policy"]
    )
    query_policy = (
        common["query_policy"] if args.mode == "original" else PAIR_QUERY_POLICY
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode", args.mode,
        "--dataset", args.dataset,
        "--budget", str(args.budget),
        "--seed", str(args.seed),
        "--max_workers", str(args.max_workers),
        "--max_tokens", str(args.max_tokens),
    ]
    if args.num_queries is not None:
        command += ["--num_queries", str(args.num_queries)]
    if args.run_tag:
        command += ["--run_tag", args.run_tag]
    if args.output is not None:
        command += ["--output", str(output)]
    if args.retrieval is not None:
        command += ["--retrieval", str(retrieval)]
    if args.image_dir is not None:
        command += ["--image_dir", str(image_dir)]
    print("Run command:", shlex.join(command))
    print("Policies:", candidate_policy, "+", query_policy)
    print("Image directory:", image_dir)
    print("Output:", output)
    if args.print_command:
        return
    if not image_dir.is_dir():
        parser.error(f"image directory not found: {image_dir}")

    core_args = [
        "--input", str(retrieval),
        "--output", str(output),
        "--checkpoint_path", str(checkpoint),
        "--checkpoint_interval_seconds", str(common["checkpoint_interval_seconds"]),
        "--schedule_log_path", str(schedule_log),
        "--image_dir", str(image_dir),
        "--api_url", args.api_url,
        "--api_key", args.api_key,
        "--model", args.model,
        "--view_size", str(common["view_size"]),
        "--top_n", str(common["top_n"]),
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(common["temperature"]),
        "--max_workers", str(args.max_workers),
        "--max_retries", str(common["max_retries"]),
        "--cand_sv", candidate_policy,
        "--cand_margin_c", str(common["candidate_margin_c"]),
        "--cand_margin_k", str(common["candidate_margin_k"]),
        "--query_sv", query_policy,
        "--budget_per_query", str(args.budget),
        "--max_per_query", str(max_per_query),
        "--batch_size", str(common["batch_size"]),
        "--shuffle_seed", str(args.seed),
        "--elo_base_rating", str(common["elo_base_rating"]),
        "--elo_rating_range", str(common["elo_rating_range"]),
        "--elo_k", str(common["elo_k"]),
        "--elo_scale", str(common["elo_scale"]),
        "--elo_max_pair_gap", str(common["elo_max_pair_gap"]),
        "--image_max_size", str(dataset["image_max_size"]),
        "--image_quality", str(dataset["image_quality"]),
    ]
    if args.num_queries is not None:
        core_args += ["--num_queries", str(args.num_queries)]

    if args.mode != "original":
        install_algorithm_hooks(common["elo_scale"])

    manifest = {
        "schema_version": 1,
        "algorithm": "amber_www_revision_v1",
        "mode": args.mode,
        "status": "running",
        "started_at_unix": time.time(),
        "ended_at_unix": None,
        "dataset": args.dataset,
        "budget_per_query": args.budget,
        "seed": args.seed,
        "num_queries": args.num_queries,
        "candidate_policy": candidate_policy,
        "query_policy": query_policy,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "algorithm_sha256": sha256(Path(__file__)),
        "retrieval_sha256": sha256(retrieval),
        "output": str(output.relative_to(ROOT)),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "schedule_log": str(schedule_log.relative_to(ROOT)),
        "parameters": {
            **common,
            "candidate_policy": candidate_policy,
            "query_policy": query_policy,
            "max_per_query": max_per_query,
            "max_workers": args.max_workers,
            "max_tokens": args.max_tokens,
            "model": args.model,
            "api_url": args.api_url,
            "image_dir": str(image_dir),
            "image_max_size": dataset["image_max_size"],
            "image_quality": dataset["image_quality"],
        },
    }
    write_json_atomic(manifest_path, manifest)
    previous_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__)), *core_args]
        run_core()
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        sys.argv = previous_argv
        manifest["ended_at_unix"] = time.time()
        write_json_atomic(manifest_path, manifest)



if __name__ == "__main__":
    main()
