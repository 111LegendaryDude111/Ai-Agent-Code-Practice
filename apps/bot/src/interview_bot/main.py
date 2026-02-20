from __future__ import annotations

import asyncio
from typing import Any

from interview_common import get_settings

from interview_bot.user_repository import UserRepository, build_user_repository

LANG_CALLBACK_PREFIX = "lang:"
SUPPORTED_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("python", "Python"),
    ("go", "Go"),
    ("java", "Java"),
    ("cpp", "C++"),
)
LANGUAGE_LABELS = dict(SUPPORTED_LANGUAGES)
MAX_SUBMISSION_LENGTH = 4000


def _import_aiogram() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from aiogram import Bot, Dispatcher, F, Router
        from aiogram.filters import CommandStart
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
    except ImportError as exc:
        raise RuntimeError(
            "aiogram is not installed. Run `make install` to install bot dependencies."
        ) from exc

    return Bot, Dispatcher, Router, CommandStart, F, InlineKeyboardButton, InlineKeyboardMarkup


def _language_keyboard() -> Any:
    _, _, _, _, _, InlineKeyboardButton, InlineKeyboardMarkup = _import_aiogram()
    buttons = [
        [
            InlineKeyboardButton(
                text=language_label,
                callback_data=f"{LANG_CALLBACK_PREFIX}{language_code}",
            )
        ]
        for language_code, language_label in SUPPORTED_LANGUAGES
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _validate_submission_text(text: str | None) -> tuple[str | None, str | None]:
    if text is None:
        return None, "Отправьте код в текстовом сообщении."

    normalized_text = text.strip()
    if normalized_text == "":
        return (
            None,
            "Пустой ввод. Отправьте непустой фрагмент кода.",
        )

    if len(normalized_text) > MAX_SUBMISSION_LENGTH:
        return (
            None,
            f"Слишком большой фрагмент кода. Максимум {MAX_SUBMISSION_LENGTH} символов.",
        )

    return normalized_text, None


def _build_dispatcher(user_repository: UserRepository) -> Any:
    _, Dispatcher, Router, CommandStart, F, _, _ = _import_aiogram()

    router = Router()
    settings = get_settings()
    submission_rate_limit_count = max(1, settings.submission_rate_limit_count)
    submission_rate_limit_window_seconds = max(
        1,
        settings.submission_rate_limit_window_seconds,
    )

    @router.message(CommandStart())
    async def start_handler(message: Any) -> None:
        from_user = message.from_user
        if from_user is None:
            await message.answer("Не удалось определить ваш Telegram ID.")
            return

        is_new_user = await user_repository.register_user(from_user.id)
        preferred_language = await user_repository.get_preferred_language(from_user.id)
        if preferred_language is None:
            status_text = "Регистрация завершена." if is_new_user else "Профиль найден."
            await message.answer(
                f"{status_text} Чтобы начать, выберите язык программирования:",
                reply_markup=_language_keyboard(),
            )
            return

        selected_language = LANGUAGE_LABELS.get(preferred_language, preferred_language)
        await message.answer(
            f"Текущий язык: {selected_language}. Можно выбрать другой:",
            reply_markup=_language_keyboard(),
        )

    @router.callback_query(F.data.startswith(LANG_CALLBACK_PREFIX))
    async def select_language_handler(callback_query: Any) -> None:
        from_user = callback_query.from_user
        if from_user is None:
            await callback_query.answer(
                "Не удалось определить пользователя.",
                show_alert=True,
            )
            return

        callback_data = callback_query.data
        if callback_data is None:
            await callback_query.answer("Некорректный выбор.", show_alert=True)
            return

        language_code = callback_data.removeprefix(LANG_CALLBACK_PREFIX)
        language_label = LANGUAGE_LABELS.get(language_code)
        if language_label is None:
            await callback_query.answer("Неизвестный язык.", show_alert=True)
            return

        await user_repository.register_user(from_user.id)
        current_language = await user_repository.get_preferred_language(from_user.id)
        if current_language == language_code:
            await callback_query.answer("Этот язык уже выбран.")
            return

        await user_repository.set_preferred_language(from_user.id, language_code)

        if current_language is None:
            response_text = f"Язык сохранен: {language_label}. Теперь можно продолжить."
        else:
            response_text = f"Язык обновлен: {language_label}."

        await callback_query.answer("Готово.")
        if callback_query.message is not None:
            await callback_query.message.answer(response_text)

    @router.message()
    async def language_required_handler(message: Any) -> None:
        from_user = message.from_user
        if from_user is None:
            return

        if message.text is not None and message.text.startswith("/"):
            return

        await user_repository.register_user(from_user.id)
        preferred_language = await user_repository.get_preferred_language(from_user.id)
        if preferred_language is not None:
            submission_text, validation_error = _validate_submission_text(message.text)
            if validation_error is not None:
                await message.answer(validation_error)
                return

            if submission_text is None:
                await message.answer("Не удалось обработать сообщение.")
                return

            recent_submissions = await user_repository.count_recent_submissions(
                from_user.id,
                submission_rate_limit_window_seconds,
            )
            if recent_submissions >= submission_rate_limit_count:
                await message.answer(
                    "Слишком много отправок за короткий интервал. "
                    f"Лимит: {submission_rate_limit_count} за "
                    f"{submission_rate_limit_window_seconds} сек."
                )
                return

            is_saved = await user_repository.save_submission(from_user.id, submission_text)
            if not is_saved:
                await message.answer(
                    "Не удалось сохранить решение. Выберите язык и повторите попытку.",
                    reply_markup=_language_keyboard(),
                )
                return

            language_label = LANGUAGE_LABELS.get(preferred_language, preferred_language)
            await message.answer(f"Код сохранен в submissions. Текущий язык: {language_label}.")
            return

        await message.answer(
            "Перед отправкой решений выберите язык программирования.",
            reply_markup=_language_keyboard(),
        )

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher


async def run_bot() -> None:
    settings = get_settings()
    if settings.bot_token is None:
        print("BOT_TOKEN is not set. Configure .env before running the Telegram bot.")
        return

    try:
        user_repository = build_user_repository(settings.database_url)
    except (RuntimeError, ValueError) as exc:
        print(f"Bot startup failed: {exc}")
        return

    try:
        await user_repository.ensure_schema()
        Bot, _, _, _, _, _, _ = _import_aiogram()
        bot = Bot(token=settings.bot_token)
        dispatcher = _build_dispatcher(user_repository)
    except RuntimeError as exc:
        print(f"Bot startup failed: {exc}")
        return

    print("Bot service started. Polling Telegram updates.")
    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()


def main() -> None:
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("Bot service interrupted by user.")


if __name__ == "__main__":
    main()
