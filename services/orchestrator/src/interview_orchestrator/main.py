from __future__ import annotations

import json
import logging

from interview_common import get_settings


def _configure_logging(log_level: str) -> None:
    normalized_level = log_level.strip().upper()
    level = getattr(logging, normalized_level, logging.INFO)
    root_logger = logging.getLogger()
    if len(root_logger.handlers) == 0:
        logging.basicConfig(level=level, format="%(message)s")
    else:
        root_logger.setLevel(level)


def main() -> None:
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger("interview_orchestrator.bootstrap")
    logger.info(
        json.dumps(
            {
                "event": "orchestrator.bootstrap",
                "app_env": settings.app_env,
                "log_level": settings.log_level,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    print(
        f"Orchestrator bootstrap complete (env={settings.app_env}, log_level={settings.log_level})."
    )


if __name__ == "__main__":
    main()
