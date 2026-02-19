from __future__ import annotations

from interview_common import get_settings


def main() -> None:
    settings = get_settings()
    if settings.bot_token is None:
        print("BOT_TOKEN is not set. Configure .env before running the Telegram bot.")
        return

    print("Bot service configuration is valid. Ready to start polling.")


if __name__ == "__main__":
    main()
