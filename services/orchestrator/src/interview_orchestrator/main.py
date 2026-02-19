from __future__ import annotations

from interview_common import get_settings


def main() -> None:
    settings = get_settings()
    print(
        f"Orchestrator bootstrap complete (env={settings.app_env}, log_level={settings.log_level})."
    )


if __name__ == "__main__":
    main()
