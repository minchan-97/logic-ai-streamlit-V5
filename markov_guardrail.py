"""
markov_guardrail.py — 도메인 자연스러움 가드레일 (v5 옵션)
==========================================================

오늘 본인이 만든 발상의 모듈화:
  "마르코프의 한계(학습 데이터 갇힘)를 가드레일에서 장점(도메인 신호기)으로"

핵심:
  - 사용자가 자기 도메인 코퍼스 텍스트를 업로드
  - 마르코프 n-gram 모델 학습 (보통 n=2)
  - LLM 응답에 마르코프 logP 점수 매김
  - 점수가 너무 낮으면 "이 응답은 학습된 도메인 흐름에서 벗어남" 신호

정직한 한계 (UI에도 표시):
  - 이건 환각 탐지기가 아니라 "도메인 자연스러움 점수기"
  - "강아지 다리 6개"처럼 표면 자연스러운 사실 오류는 못 잡음
  - 학습 코퍼스가 작으면 가짜 negative 많음 (자연스러운 응답을 거부)
  - LLM의 일반 어휘가 자기 도메인 코퍼스 어휘와 다르면 거의 다 -10점

v4와 보완:
  - v4 mismatch: LLM 자체 헷갈림 신호
  - 마르코프 점수: 도메인 밖 표현 신호
  - 둘은 다른 종류의 신호 (실험 5에서 확인)
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class GuardrailConfig:
    """마르코프 가드레일 설정."""
    n: int = 2                      # n-gram 크기
    unknown_penalty: float = -10.0  # 학습 안 된 context 패널티
    warning_threshold: float = -5.0  # 이 값보다 logP 낮으면 경고
    min_corpus_tokens: int = 100   # 코퍼스가 이보다 작으면 학습 안 함
    enabled: bool = False           # 기본 OFF


@dataclass
class GuardrailState:
    """학습된 마르코프 모델 + 메타."""
    model: Dict = field(default_factory=dict)
    vocab: List[str] = field(default_factory=list)
    corpus_tokens: int = 0
    is_trained: bool = False
    corpus_name: str = ""


def tokenize_text(text: str) -> List[str]:
    """단순 어절 토큰화."""
    return [t for t in text.replace("\n", " ").split() if t]


def train_guardrail(text: str, cfg: GuardrailConfig) -> GuardrailState:
    """텍스트로 마르코프 가드레일 학습.
    
    args:
        text: 도메인 코퍼스 텍스트 (충분히 큰)
        cfg: 설정
    
    returns:
        학습된 GuardrailState
    """
    tokens = tokenize_text(text)
    if len(tokens) < cfg.min_corpus_tokens:
        raise ValueError(
            f"코퍼스가 너무 작습니다 ({len(tokens)} 토큰). "
            f"최소 {cfg.min_corpus_tokens} 토큰 필요."
        )
    
    n = cfg.n
    model: Dict[Tuple, Counter] = defaultdict(Counter)
    for i in range(len(tokens) - n):
        context = tuple(tokens[i:i + n])
        next_tok = tokens[i + n]
        model[context][next_tok] += 1
    
    state = GuardrailState(
        model=dict(model),
        vocab=sorted(set(tokens)),
        corpus_tokens=len(tokens),
        is_trained=True,
    )
    return state


def score_response(response: str, state: GuardrailState,
                    cfg: GuardrailConfig) -> Dict:
    """LLM 응답에 마르코프 점수 매김.
    
    returns:
        - avg_logp: 평균 log 확률 (높을수록 자연스러움)
        - log_likelihood: 전체 log P
        - coverage: 학습된 transition 비율
        - unknown_count: 학습 안 된 context 수
        - is_warning: warning_threshold 초과 여부
        - per_token: 위치별 logP 리스트 (시각화용)
    """
    if not state.is_trained:
        return {"error": "가드레일이 학습되지 않음"}
    
    n = cfg.n
    tokens = tokenize_text(response)
    
    if len(tokens) <= n:
        return {
            "avg_logp": float("nan"),
            "warning": True,
            "reason": f"응답이 너무 짧음 (n={n} 미만)"
        }
    
    per_token = []
    unknown = 0
    valid = 0
    
    for i in range(n, len(tokens)):
        context = tuple(tokens[i - n:i])
        next_tok = tokens[i]
        
        if context not in state.model:
            per_token.append({"token": next_tok, "logp": cfg.unknown_penalty,
                              "status": "unknown_context"})
            unknown += 1
            continue
        
        next_counts = state.model[context]
        total = sum(next_counts.values())
        count = next_counts.get(next_tok, 0)
        
        if count == 0:
            # context는 봤지만 next_tok 안 나옴
            prob = 1.0 / (total + len(next_counts) + 10)
            per_token.append({"token": next_tok, "logp": float(np.log(prob)),
                              "status": "unseen_transition"})
            unknown += 1
        else:
            prob = count / total
            per_token.append({"token": next_tok, "logp": float(np.log(prob)),
                              "status": "ok"})
            valid += 1
    
    logps = [p["logp"] for p in per_token]
    avg_logp = float(np.mean(logps)) if logps else float("nan")
    total_logp = float(np.sum(logps)) if logps else float("nan")
    coverage = valid / max(len(per_token), 1)
    is_warning = avg_logp < cfg.warning_threshold
    
    return {
        "avg_logp": avg_logp,
        "log_likelihood": total_logp,
        "coverage": coverage,
        "unknown_count": unknown,
        "valid_count": valid,
        "is_warning": is_warning,
        "warning_threshold": cfg.warning_threshold,
        "per_token": per_token,
        "n_tokens": len(tokens),
    }


def interpret_score(avg_logp: float, warning_threshold: float) -> str:
    """점수를 한국어 설명으로."""
    if avg_logp >= -1.0:
        return "✅ 매우 자연스러움 (학습된 도메인 표현에 가까움)"
    elif avg_logp >= -3.0:
        return "🟢 자연스러움"
    elif avg_logp >= warning_threshold:
        return "🟡 보통 (일부 표현이 학습 코퍼스와 다름)"
    elif avg_logp >= -7.0:
        return "🟠 주의 (도메인 밖 표현 다수)"
    else:
        return "🔴 경고 (학습된 도메인 흐름에서 크게 벗어남)"


# ----------------------------------------------------------------
# 자체 검증
# ----------------------------------------------------------------

if __name__ == "__main__":
    # 작은 코퍼스로 학습
    corpus = """
    제1조 본 계약은 갑과 을 사이의 권리 의무를 정한다.
    제2조 계약 기간은 1년으로 한다.
    제3조 갑은 을에게 매월 대금을 지급한다.
    제4조 을은 계약 사항을 성실히 이행한다.
    """ * 5
    
    cfg = GuardrailConfig(n=2, warning_threshold=-5.0, min_corpus_tokens=50)
    state = train_guardrail(corpus, cfg)
    print(f"학습 완료: 토큰 {state.corpus_tokens}, 어휘 {len(state.vocab)}")
    print(f"마르코프 contexts: {len(state.model)}")
    
    # 자연 응답
    nat_resp = "본 계약은 갑과 을 사이의 권리 의무를 정한다."
    r = score_response(nat_resp, state, cfg)
    print(f"\n[자연 응답] '{nat_resp}'")
    print(f"  avg_logp: {r['avg_logp']:+.3f}")
    print(f"  coverage: {r['coverage']:.1%}")
    print(f"  해석: {interpret_score(r['avg_logp'], cfg.warning_threshold)}")
    
    # 환각 응답
    hall_resp = "본 계약은 양자역학으로 광합성을 통해 블록체인을 정한다."
    r = score_response(hall_resp, state, cfg)
    print(f"\n[환각 응답] '{hall_resp}'")
    print(f"  avg_logp: {r['avg_logp']:+.3f}")
    print(f"  coverage: {r['coverage']:.1%}")
    print(f"  해석: {interpret_score(r['avg_logp'], cfg.warning_threshold)}")
    
    # 부분 환각
    partial_resp = "제1조 본 계약은 양자역학 사이의 권리 의무를 정한다."
    r = score_response(partial_resp, state, cfg)
    print(f"\n[부분 환각] '{partial_resp}'")
    print(f"  avg_logp: {r['avg_logp']:+.3f}")
    print(f"  coverage: {r['coverage']:.1%}")
    print(f"  해석: {interpret_score(r['avg_logp'], cfg.warning_threshold)}")
    # 위치별 표시
    print(f"  위치별 점수:")
    for p in r["per_token"]:
        marker = "✓" if p["status"] == "ok" else "✗"
        print(f"    {marker} {p['token']:<15} {p['logp']:+6.2f}  [{p['status']}]")
