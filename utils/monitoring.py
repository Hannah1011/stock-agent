"""
utils/monitoring.py

AgentLog 수집·조회·포맷팅 모듈.

역할:
  1. 워크플로우 실행 중 생성된 AgentLog를 스레드 안전하게 수집
  2. Streamlit UI의 "에이전트 사고 과정" 섹션에 표시할 형태로 포맷팅
  3. 실행 통계(성공/오류/스킵 수) 요약

사용 패턴 (Streamlit):
  monitor = MonitoringService()
  result  = workflow.run(user_input, on_log=monitor.add)
  st.markdown(monitor.format_for_expander())

로그 표시 예시:
  Orchestrator (Sonnet)    [RUNNING]  의도 분류 중…
  Orchestrator (Sonnet)    [DONE]     intent=A | 삼성전자        (1.2초)
  NewsCollector (Haiku)    [RUNNING]  뉴스 수집 중…
  ChartAnalyst (Haiku)     [RUNNING]  차트 분석 중…
  NewsCollector (Haiku)    [DONE]     5건 수집 완료              (2.8초)
  ChartAnalyst (Haiku)     [DONE]     005930.KS 분석 완료        (3.1초)
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

from schemas.models import AgentLog

# 상태 배지 텍스트 (마크다운 인라인 코드 블록으로 표시)
_STATUS_BADGES: dict[str, str] = {
    "running": "RUNNING",
    "done":    "DONE",
    "error":   "ERROR",
    "skipped": "SKIP",
}

# 사람이 읽기 쉬운 모델명 축약
_MODEL_LABELS: dict[str, str] = {
    "claude-sonnet-4-6":         "Sonnet",
    "claude-haiku-4-5-20251001": "Haiku",
}


class MonitoringService:
    """
    에이전트 실행 로그를 수집하고 Streamlit 표시용으로 포맷팅한다.
    스레드 안전하게 설계되어 병렬 에이전트 실행 시에도 안전하다.

    Streamlit에서는 session_state에 인스턴스를 저장해 세션별로 분리한다.
    (@st.cache_resource는 세션 간 공유되므로 로그 혼용 문제가 생길 수 있음)
    """

    def __init__(self) -> None:
        self._logs: list[AgentLog] = []
        self._lock = threading.Lock()

    # ── 로그 추가 ────────────────────────────────────────────────────────────

    def add(self, log: AgentLog) -> None:
        """로그를 수집 목록에 추가한다. on_log 콜백으로 직접 사용 가능하다."""
        with self._lock:
            self._logs.append(log)

    def add_all(self, logs: list[AgentLog]) -> None:
        """로그 목록을 한꺼번에 추가한다. 워크플로우 완료 후 일괄 동기화 시 사용한다."""
        with self._lock:
            self._logs.extend(logs)

    # ── 조회 ─────────────────────────────────────────────────────────────────

    def get_all(self) -> list[AgentLog]:
        """현재까지 수집된 모든 로그의 복사본을 반환한다."""
        with self._lock:
            return list(self._logs)

    def count(self) -> int:
        """수집된 로그 수를 반환한다."""
        with self._lock:
            return len(self._logs)

    def clear(self) -> None:
        """수집된 모든 로그를 초기화한다. 새 대화 시작 시 호출한다."""
        with self._lock:
            self._logs = []

    def has_error(self) -> bool:
        """오류가 발생한 에이전트가 하나 이상 있으면 True를 반환한다."""
        return any(log.status == "error" for log in self.get_all())

    # ── Streamlit 표시용 포맷팅 ─────────────────────────────────────────────

    def format_for_expander(self) -> str:
        """
        Streamlit expander("에이전트 사고 과정")에 표시할 마크다운 텍스트를 반환한다.

        형식:
          AgentName (Model)    [STATUS]   action 내용              (소요시간)
          AgentName (Model)    [ERROR]    action 내용
                                         > ERROR 메시지

        DONE / ERROR 항목에는 대응하는 RUNNING 로그 기준 소요 시간을 표시한다.
        에이전트 이름 컬럼은 24자로 고정 패딩해 정렬을 맞춘다.
        """
        logs = self.get_all()
        if not logs:
            return "_실행 로그가 없습니다._"

        # 에이전트별 가장 최근 RUNNING 타임스탬프 추적
        running_times: dict[str, str] = {}
        lines: list[str] = []

        for log in logs:
            badge       = _STATUS_BADGES.get(log.status, "?")
            model_label = _MODEL_LABELS.get(log.model_used or "", "")
            model_info  = f" ({model_label})" if model_label else ""

            if log.status == "running":
                running_times[log.agent] = log.timestamp
                elapsed_str = ""
            else:
                start_ts    = running_times.get(log.agent)
                elapsed_str = self._format_elapsed(start_ts, log.timestamp) if start_ts else ""

            line = f"`{badge}`  **{log.agent}**{model_info}  {log.action}{elapsed_str}"

            if log.status == "error" and log.error:
                line += f"\n\n> {log.error}"

            lines.append(line)

        return "\n\n".join(lines)

    def format_summary(self) -> str:
        """
        에이전트 실행 통계를 한 줄 요약으로 반환한다.

        각 에이전트의 '마지막 상태'만 집계한다.
        (running → done 순서로 로그가 쌓이므로 done이 최종 상태)

        예: "에이전트 5개 실행 — DONE 4 / ERROR 1 / SKIP 0"
        """
        logs = self.get_all()
        if not logs:
            return "실행 기록 없음"

        last_status: dict[str, str] = {}
        for log in logs:
            last_status[log.agent] = log.status

        total   = len(last_status)
        done    = sum(1 for s in last_status.values() if s == "done")
        errors  = sum(1 for s in last_status.values() if s == "error")
        skipped = sum(1 for s in last_status.values() if s == "skipped")

        return f"에이전트 {total}개 실행 — DONE {done} / ERROR {errors} / SKIP {skipped}"

    # ── 정적 헬퍼 ────────────────────────────────────────────────────────────

    @staticmethod
    def _format_elapsed(start_ts: str, end_ts: str) -> str:
        """
        두 ISO 8601 타임스탬프 사이의 경과 시간을 "(N.N초)" 형태로 반환한다.
        60초 이상이면 "(N.N분)"으로 표시한다.
        파싱 실패 시 빈 문자열을 반환한다.
        """
        try:
            start   = datetime.fromisoformat(start_ts)
            end     = datetime.fromisoformat(end_ts)
            elapsed = (end - start).total_seconds()
            if elapsed < 0:
                return ""
            if elapsed < 60:
                return f"  ({elapsed:.1f}초)"
            return f"  ({elapsed / 60:.1f}분)"
        except (ValueError, TypeError, OSError):
            return ""

    @staticmethod
    def make_log(
        agent: str,
        status: str,
        action: str,
        model_used: Optional[str] = None,
        error: Optional[str] = None,
    ) -> AgentLog:
        """
        AgentLog 인스턴스를 생성한다 (수집 목록에는 추가하지 않음).
        워크플로우 외부에서 수동으로 로그를 생성할 때 사용한다.
        """
        return AgentLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=agent,
            status=status,
            action=action,
            model_used=model_used,
            error=error,
        )
