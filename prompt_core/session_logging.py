import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


def get_logs_dir() -> Path:
    logs_dir = Path.home() / ".prompt-core" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def log_session(
    messages: list[dict[str, str]],
    criteria: dict[str, Any] | None,
    success_judgement: bool,
    feedback_text: str | None,
    model: str,
    turn_count: int,
    context: str,
) -> Path:
    logs_dir = get_logs_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"session_{timestamp}.json"
    log_path = logs_dir / log_filename

    session_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "turn_count": turn_count,
        "context": context,
        "messages": messages,
        "criteria": criteria,
        "user_feedback": {
            "success_judgement": success_judgement,
            "feedback_text": feedback_text,
        },
    }

    handler_id = logger.add(log_path, level="INFO", format="{message}")
    try:
        logger.info(json.dumps(session_data, indent=2))
    finally:
        logger.remove(handler_id)

    logger.info("Session logged to {}", log_path)
    return log_path
