import os
import json
import threading
import time
from datetime import datetime

from config import ANALYTICS_FILE, MAX_EVENTS

_lock = threading.Lock()


def _load() -> list:
    if not os.path.exists(ANALYTICS_FILE):
        return []
    try:
        with open(ANALYTICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(events: list):
    tmp = ANALYTICS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False)
    os.replace(tmp, ANALYTICS_FILE)


def log_event(event_type: str, **payload):
    """Records one usage/health event. Safe to call from worker threads."""
    try:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": event_type,
            **payload,
        }
        with _lock:
            events = _load()
            events.append(entry)
            if len(events) > MAX_EVENTS:
                events = events[-MAX_EVENTS:]
            _save(events)
    except Exception:
        pass


def log_query(query, role=None, machine=None, subsystem=None, intent=None,
              latency_ms=None, retrieval_score=None, sources=None,
              vision=False, code_match=None, session_id=None, mode=None, status="ok"):
    """Convenience wrapper for logging a chat query."""
    log_event(
        "query",
        query=query[:200],
        role=role,
        machine=machine,
        subsystem=subsystem,
        intent=intent,
        latency_ms=round(latency_ms, 1) if latency_ms is not None else None,
        retrieval_score=round(retrieval_score, 4) if retrieval_score is not None else None,
        sources=sources or [],
        vision=vision,
        code_match=code_match,
        session_id=session_id,
        mode=mode,
        status=status,
    )


def log_system(model=None, device=None, threads=None, ctx=None, vision_loaded=False,
               chunks=0, status="ok"):
    log_event(
        "system",
        model=model,
        device=device,
        threads=threads,
        ctx=ctx,
        vision_loaded=vision_loaded,
        chunks=chunks,
        status=status,
    )


def get_stats() -> dict:
    """Aggregates lightweight KPIs over the event log."""
    events = _load()
    total = len(events)
    queries = [e for e in events if e.get("type") == "query"]
    q_count = len(queries)
    ok_queries = [q for q in queries if q.get("status") == "ok"]
    latencies = [q["latency_ms"] for q in ok_queries if q.get("latency_ms") is not None]
    vis_count = sum(1 for q in queries if q.get("vision"))
    code_count = sum(1 for q in queries if q.get("code_match"))
    machines = {}
    for q in queries:
        m = q.get("machine") or "Unspecified"
        machines[m] = machines.get(m, 0) + 1

    return {
        "total_events": total,
        "total_queries": q_count,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "vision_queries": vis_count,
        "fault_code_queries": code_count,
        "top_machines": sorted(machines.items(), key=lambda x: x[1], reverse=True)[:10],
        "success_rate": round((len(ok_queries) / q_count) * 100, 1) if q_count else None,
    }


def export_csv(path: str) -> str:
    """Writes the event log to a CSV for management reporting. Returns written path."""
    import csv

    events = _load()
    if not events:
        raise ValueError("No analytics events to export.")

    fields = sorted({k for e in events for k in e.keys()})
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in events:
            row = {k: e.get(k, "") for k in fields}
            if isinstance(row.get("sources"), (list, dict)):
                row["sources"] = json.dumps(row["sources"], ensure_ascii=False)
            writer.writerow(row)
    return path


def reset():
    with _lock:
        _save([])


def timer_start():
    return time.perf_counter()


def timer_ms(start):
    return (time.perf_counter() - start) * 1000.0