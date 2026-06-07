"""
state_path.py — 마르코프 상태-경로 추적 (v5 옵션 모듈)
========================================================

기존 markov_guardrail.py는 응답에 평균 logP 한 점수 매김.
이 모듈은 그 위에 '위치별 추적' 능력을 더함:
  - 각 토큰 위치마다 마르코프 context가 학습 그래프 안인가 밖인가
  - 첫 점프 위치 (환각 시작점 추정)
  - 토큰별 진단 (어떤 단어가 어긋났는가)

기존 가드레일과 분리: import해서 함께 쓰지만, 단독으로도 작동.

실험 7에서 검증된 결과 (시뮬레이션):
  - 진짜 응답: inside_ratio 100%, 점프 없음 100%
  - 외부 환각: inside_ratio 86%, 점프 발생 53%
  - 부분 환각: inside_ratio 90%, 점프 발생 73%
  - 첫 점프 위치 = 환각 단어 삽입 위치 거의 일치

정직한 한계:
  - "그래프 안/밖" 임계값 임의 (코퍼스마다 조정 필요)
  - context 좌표 = 단어 임베딩 평균 (n>2면 정확도 감소)
  - inside_ratio 단독으론 약함, "점프 발생 여부"가 진짜 신호
  - 시뮬레이션 검증만 됨. 진짜 LLM 환각에선 다를 수 있음
"""
from __future__ import annotations
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class StatePathConfig:
    """위치별 추적 설정."""
    threshold_ratio: float = 0.10  # 학습 천이 평균 거리의 X% = "근처" 임계값
    enabled: bool = False           # 기본 OFF


@dataclass
class StatePathState:
    """학습된 천이 그래프 + 임베딩 좌표."""
    transitions: List[Tuple[np.ndarray, np.ndarray, int]] = field(default_factory=list)
    coords_2d: Optional[np.ndarray] = None
    word2idx: Dict[str, int] = field(default_factory=dict)
    threshold: float = 0.0
    n: int = 2
    is_built: bool = False


def tokenize_simple(text: str) -> List[str]:
    """단순 어절 토큰화."""
    return [t for t in text.replace("\n", " ").split() if t]


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """numpy PCA 2D 환원."""
    X = vectors - vectors.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def context_to_coord(context: Tuple[str, ...], coords_2d: np.ndarray,
                       word2idx: Dict[str, int]) -> Optional[np.ndarray]:
    """context (단어 n개)를 한 좌표로 매핑. 구성 단어 임베딩 평균."""
    valid = [coords_2d[word2idx[w]] for w in context if w in word2idx]
    if not valid:
        return None
    return np.mean(valid, axis=0)


def build_state_path_state(
    text: str,
    n: int = 2,
    embedding_dim: int = 16,
    embedding_epochs: int = 15,
    config: Optional[StatePathConfig] = None,
) -> StatePathState:
    """텍스트로 임베딩 + 마르코프 학습 후 상태 그래프 구축.
    
    가드레일과 같은 코퍼스를 쓰지만 임베딩이 추가로 필요해서 따로 학습.
    """
    cfg = config or StatePathConfig()
    
    # Skip-gram 임베딩 (Layer B와 같은 형태, 의존성 없이 여기 내장)
    tokens = tokenize_simple(text)
    if len(tokens) < 50:
        raise ValueError(f"코퍼스가 너무 작음 ({len(tokens)} 토큰)")
    
    # 어휘
    counter = Counter(tokens)
    words = [w for w, c in counter.items() if c >= 2]
    word2idx = {w: i for i, w in enumerate(words)}
    if len(words) < 10:
        raise ValueError(f"어휘가 너무 작음 ({len(words)})")
    
    # Skip-gram 학습 (간단 버전, 의존성 0)
    rng = np.random.default_rng(42)
    scale = 1.0 / embedding_dim
    W_in = (rng.random((len(words), embedding_dim)) - 0.5) / embedding_dim
    W_out = (rng.random((len(words), embedding_dim)) - 0.5) / embedding_dim
    lr = 0.1
    
    # 학습 쌍
    window = 2
    pairs = []
    for i, w in enumerate(tokens):
        if w not in word2idx:
            continue
        c = word2idx[w]
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if j == i or tokens[j] not in word2idx:
                continue
            pairs.append((c, word2idx[tokens[j]]))
    
    def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    for ep in range(embedding_epochs):
        rng.shuffle(pairs)
        for center, ctx in pairs:
            negs = rng.integers(0, len(words), size=5)
            v_c = W_in[center]
            v_pos = W_out[ctx]
            p_pos = sig(np.dot(v_c, v_pos))
            grad_c = (p_pos - 1.0) * v_pos
            for neg in negs:
                v_neg = W_out[neg]
                p_neg = sig(np.dot(v_c, v_neg))
                grad_c += p_neg * v_neg
                W_out[neg] -= lr * p_neg * v_c
            W_out[ctx] -= lr * (p_pos - 1.0) * v_c
            W_in[center] -= lr * grad_c
    
    coords_2d = pca_2d(W_in)
    
    # 마르코프 모델 (가드레일과 같은 방식)
    model = defaultdict(Counter)
    for i in range(len(tokens) - n):
        context = tuple(tokens[i:i + n])
        next_tok = tokens[i + n]
        model[context][next_tok] += 1
    
    # 천이 그래프
    transitions = []
    for context, next_counts in model.items():
        from_c = context_to_coord(context, coords_2d, word2idx)
        if from_c is None:
            continue
        for next_word, count in next_counts.items():
            new_context = context[1:] + (next_word,)
            to_c = context_to_coord(new_context, coords_2d, word2idx)
            if to_c is None:
                continue
            transitions.append((from_c, to_c, count))
    
    # 임계값 결정: 학습 천이 from_coord들의 평균 인접 거리의 threshold_ratio
    from_arr = np.array([t[0] for t in transitions])
    if len(from_arr) < 2:
        threshold = 0.1
    else:
        sample_size = min(100, len(from_arr))
        idx = rng.choice(len(from_arr), size=sample_size * 2, replace=True)
        dists = [
            float(np.linalg.norm(from_arr[idx[i]] - from_arr[idx[i + sample_size]]))
            for i in range(sample_size)
        ]
        threshold = float(np.mean(dists)) * cfg.threshold_ratio
    
    return StatePathState(
        transitions=transitions,
        coords_2d=coords_2d,
        word2idx=word2idx,
        threshold=threshold,
        n=n,
        is_built=True,
    )


def is_state_in_graph(coord: np.ndarray, transitions: List,
                       threshold: float) -> bool:
    if not transitions:
        return False
    from_coords = np.array([t[0] for t in transitions])
    dists = np.linalg.norm(from_coords - coord, axis=1)
    return float(np.min(dists)) < threshold


def analyze_response_path(response: str, state: StatePathState) -> Dict:
    """응답의 위치별 추적 분석.
    
    반환:
      - per_position: 각 위치별 (context, in_graph) 리스트
      - first_jump_position: 처음 점프 위치 (-1이면 점프 없음)
      - inside_ratio: 그래프 안 비율
      - jump_count: 점프 횟수
      - jump_tokens: 점프가 일어난 단어들
    """
    if not state.is_built:
        return {"error": "상태 경로 모델 학습 안 됨"}
    
    tokens = tokenize_simple(response)
    if len(tokens) < state.n + 1:
        return {"error": "응답이 너무 짧음"}
    
    per_position = []
    for i in range(state.n, len(tokens) + 1):
        context = tuple(tokens[i - state.n:i])
        coord = context_to_coord(context, state.coords_2d, state.word2idx)
        if coord is None:
            per_position.append({
                "position": i - state.n,
                "context": context,
                "in_graph": False,
                "reason": "어휘 밖 단어",
            })
        else:
            in_g = is_state_in_graph(coord, state.transitions, state.threshold)
            per_position.append({
                "position": i - state.n,
                "context": context,
                "in_graph": in_g,
                "reason": "ok" if in_g else "그래프 밖",
            })
    
    if not per_position:
        return {"error": "분석 가능한 위치 없음"}
    
    in_graph_flags = [p["in_graph"] for p in per_position]
    inside_ratio = sum(in_graph_flags) / len(in_graph_flags)
    
    # 첫 점프
    first_jump = -1
    for i, in_g in enumerate(in_graph_flags):
        if not in_g:
            first_jump = i
            break
    
    # 점프 횟수
    jump_count = sum(1 for in_g in in_graph_flags if not in_g)
    
    # 점프 자리의 단어들
    jump_tokens = []
    for p in per_position:
        if not p["in_graph"]:
            # context의 마지막 단어가 새로 등장한 단어
            jump_tokens.append(p["context"][-1])
    
    return {
        "per_position": per_position,
        "first_jump_position": first_jump,
        "inside_ratio": inside_ratio,
        "jump_count": jump_count,
        "n_positions": len(per_position),
        "jump_tokens": jump_tokens,
    }


def interpret_path(analysis: Dict) -> str:
    """분석을 한국어 한 줄로."""
    if "error" in analysis:
        return f"⚠️ 분석 불가: {analysis['error']}"
    
    inside = analysis["inside_ratio"]
    first_jump = analysis["first_jump_position"]
    jump_count = analysis["jump_count"]
    
    if first_jump == -1:
        return "✅ 학습 그래프 안에서 안정적으로 흐름 (점프 없음)"
    elif jump_count == 1:
        return f"🟡 위치 {first_jump}에서 1회 점프 (일부 도메인 밖 표현)"
    elif jump_count <= 3:
        return f"🟠 {jump_count}회 점프 (첫 점프 위치 {first_jump})"
    else:
        return f"🔴 {jump_count}회 다수 점프 (첫 점프 {first_jump}) — 도메인 흐름 크게 이탈"


if __name__ == "__main__":
    # 자체 검증
    corpus = """
    제1조 본 계약은 갑과 을 사이의 권리 의무를 정한다.
    제2조 계약 기간은 1년으로 한다.
    제3조 갑은 을에게 매월 대금을 지급한다.
    제4조 을은 계약 사항을 성실히 이행한다.
    제5조 본 계약은 양 당사자가 서명한 날부터 효력이 발생한다.
    제6조 계약 변경은 서면 합의로만 가능하다.
    """ * 8
    
    print("상태 경로 학습 중...")
    state = build_state_path_state(corpus, n=2)
    print(f"  학습 천이: {len(state.transitions)}, 어휘: {len(state.word2idx)}")
    print(f"  임계값: {state.threshold:.4f}")
    
    # 자연
    nat = "본 계약은 갑과 을 사이의 권리 의무를 정한다."
    a = analyze_response_path(nat, state)
    print(f"\n[자연] '{nat}'")
    print(f"  {interpret_path(a)}")
    print(f"  inside_ratio: {a['inside_ratio']:.1%}, 첫 점프: {a['first_jump_position']}")
    
    # 환각
    hall = "본 계약은 양자역학으로 광합성을 통해 블록체인을 정한다."
    a = analyze_response_path(hall, state)
    print(f"\n[환각] '{hall}'")
    print(f"  {interpret_path(a)}")
    print(f"  inside_ratio: {a['inside_ratio']:.1%}, 첫 점프: {a['first_jump_position']}")
    print(f"  점프 단어들: {a['jump_tokens']}")
    
    # 부분 환각
    partial = "제1조 본 계약은 양자역학 사이의 권리 의무를 정한다."
    a = analyze_response_path(partial, state)
    print(f"\n[부분 환각] '{partial}'")
    print(f"  {interpret_path(a)}")
    print(f"  inside_ratio: {a['inside_ratio']:.1%}, 첫 점프: {a['first_jump_position']}")
    print(f"  점프 단어들: {a['jump_tokens']}")
    print(f"  위치별:")
    for p in a["per_position"]:
        marker = "✓" if p["in_graph"] else "✗"
        ctx_str = " ".join(p["context"])
        print(f"    [{p['position']}] {marker} ({ctx_str})")
