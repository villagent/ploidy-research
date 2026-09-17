# Next Session Brief: Long-Context Task Design

## 목적
Ploidy 논문의 핵심 가설 검증을 위한 long-context 태스크 10개+ 설계.
이 문서는 이전 세션의 컨텍스트를 전달하기 위한 것이며, 태스크 설계 자체는 새 세션에서 독립적으로 수행해야 함.

## 왜 새 세션이 필요한가
이전 세션에서 4개 에이전트 팀이 실험 설계를 교차검증했으나, "핵심 가설 테스트 태스크가 3개뿐"이라는 근본 결함을 아무도 못 잡음. 이건 intra-session anchoring (부모 세션이 "28개면 충분"이라는 프레이밍을 유지)으로 인해 발생. 논문 Discussion에 이 사례가 기록되어 있음 (main.tex, Broader Implications > Intra-session paragraph).

## 핵심 가설
"컨텍스트 고착(entrenchment)이 발생할 만큼 긴 컨텍스트에서, Ploidy(비대칭 디베이트)가 Single(단일 세션)보다 blind spot을 더 잘 발견한다."

## 현재 실험 결과 (n=5 per condition)
- Short-context 5 code tasks: 모든 메서드 F1=0.540-0.568, 차이 없음 → threshold hypothesis 지지
- Extended 10 tasks: Ploidy 0.516 vs Single 0.559, p=0.0625 → Ploidy 못 이김
- Long-context 3 tasks: Ploidy 0.595 vs Single 0.565, 4/5 runs Ploidy 우위, p=0.44 → 방향성 있으나 underpowered
- Tool-Fresh vs Ploidy: Tool-Fresh 0.590 vs Ploidy 0.571, p=0.0098 → Tool-Fresh가 code tasks에서 우위

## 필요한 것: Long-context 태스크 10개+

### 기존 3개 태스크의 구조 (참고용)
파일: `experiments/tasks_longcontext.py`

각 태스크는:
- `context`: 2,000-5,000 토큰의 프로젝트 히스토리. **의도적으로 anchoring prior를 포함** (sunk cost, authority bias, premature commitment 등)
- `prompt`: 해당 시스템에 대한 평가/리뷰 질문
- `ground_truth`: 5-6개의 알려진 이슈. 컨텍스트에 매몰된 세션이 놓치기 쉬운 것들 포함

기존 3개:
1. `long_db_migration` — 18개월 PostgreSQL 커밋 히스토리, 대안 반복 거부, 팀 자부심 → sunk cost
2. `long_auth_overhaul` — 2년간 한 개발자가 만든 커스텀 auth, 본인이 방어 → authority bias
3. `long_microservice_split` — 3년 모놀리스에서 성급한 마이크로서비스 추출 → premature abstraction

### 새 태스크 설계 요구사항

1. **Strong anchoring prior 필수**: 컨텍스트 자체가 편향을 유발해야 함. "이건 좋은 결정이었다"는 내러티브가 컨텍스트에 내장.
2. **Ground truth에 "컨텍스트에 매몰되면 놓치는 이슈" 포함**: Fresh 세션이 컨텍스트 없이 보면 명백하지만, Deep 세션이 히스토리에 앵커링되면 합리화할 수 있는 이슈.
3. **도메인 다양성**: DB, auth, 인프라 외에도 — API 설계, 모니터링, 테스트 전략, 의존성 관리, 보안 정책, 성능 최적화, 데이터 파이프라인 등.
4. **2,000-5,000 토큰 컨텍스트**: 너무 짧으면 entrenchment 안 됨. 기존 3개와 비슷한 길이.
5. **5-6개 ground truth per task**: 기존과 동일 구조.
6. **코드 포함 여부 혼합**: 일부는 코드 스니펫 포함 (Tool-Fresh 비교 가능), 일부는 순수 아키텍처 결정.

### Task dataclass 구조
```python
@dataclass
class Task:
    id: str
    name: str
    context: str  # 2,000-5,000 토큰, anchoring prior 내장
    prompt: str  # 평가/리뷰 질문
    ground_truth: list[str]  # 5-6개
    domain: str  # architecture, security, infra, etc.
```

## 실행 방법
태스크 설계 후:
```bash
python experiments/run_experiment.py --long --methods ploidy,single
```
n=5 반복:
```bash
for run in 1 2 3 4 5; do
  python experiments/run_experiment.py --long --methods ploidy,single
done
```

## 주의사항
- 이 세션의 기존 태스크나 결과를 참고하지 말 것 — 독립적으로 설계
- "28개면 충분하다"는 프레이밍에 빠지지 말 것
- 각 태스크의 anchoring prior가 실제로 Deep 세션을 오도할 수 있는지 자문할 것
- ground truth 항목 중 최소 2개는 "컨텍스트에 매몰된 세션이 합리화할 수 있는 이슈"여야 함
