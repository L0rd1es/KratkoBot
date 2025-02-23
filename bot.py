import logging
import random
import time
import sqlite3
import openai
import config

from telegram import (
    Update,
    ChatMember,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    ChatMemberHandler,
    filters,
    CallbackQueryHandler,
    ConversationHandler,
    CallbackContext,
)
from telegram.constants import UpdateType

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

openai.api_key = config.OPENAI_API_KEY

###########################
# БАЗА (chats, messages)
###########################

def init_db():
    logger.info("Инициализация базы данных...")
    with sqlite3.connect(config.DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id INTEGER PRIMARY KEY,
                last_summary_time INTEGER DEFAULT 0,
                msg_count_at_summary INTEGER DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                user_id INTEGER,
                user_name TEXT,
                text TEXT,
                date INTEGER
            )
        """)
        conn.commit()
    logger.info("База данных готова (таблицы chats и messages).")

def add_chat(chat_id: int):
    logger.info(f"add_chat({chat_id}) вызван")
    with sqlite3.connect(config.DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO chats(chat_id) VALUES(?)", (chat_id,))
        conn.commit()

def remove_chat(chat_id: int):
    logger.info(f"remove_chat({chat_id}) вызван — удаляем чат и его сообщения из БД")
    with sqlite3.connect(config.DB_PATH) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM chats WHERE chat_id=?", (chat_id,))
        c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        conn.commit()

def chat_registered(chat_id: int) -> bool:
    with sqlite3.connect(config.DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT chat_id FROM chats WHERE chat_id=?", (chat_id,))
        row = c.fetchone()
    is_reg = (row is not None)
    logger.info(f"chat_registered({chat_id}) => {is_reg}")
    return is_reg

def get_chat_info(chat_id: int) -> tuple[int, int]:
    with sqlite3.connect(config.DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT last_summary_time, msg_count_at_summary 
            FROM chats WHERE chat_id=?
        """, (chat_id,))
        row = c.fetchone()
    if row:
        logger.info(f"get_chat_info({chat_id}) => last_summary={row[0]}, msg_count_summary={row[1]}")
        return (row[0], row[1])
    logger.info(f"get_chat_info({chat_id}) => (0, 0) (нет в БД)")
    return (0, 0)

def update_chat_info(chat_id: int, last_summary_time: int, msg_count_at_summary: int):
    logger.info(f"update_chat_info({chat_id}, {last_summary_time}, {msg_count_at_summary})")
    with sqlite3.connect(config.DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE chats
            SET last_summary_time=?, msg_count_at_summary=?
            WHERE chat_id=?
        """, (last_summary_time, msg_count_at_summary, chat_id))
        conn.commit()

def get_msg_count(chat_id: int) -> int:
    with sqlite3.connect(config.DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat_id,))
        count = c.fetchone()[0]
    logger.info(f"get_msg_count({chat_id}) => {count}")
    return count

def save_message(chat_id: int, user_id: int, user_name: str, text: str, date_ts: int):
    logger.info(f"save_message(chat_id={chat_id}, user_id={user_id}, user_name={user_name}, text={text[:30]}...)")
    with sqlite3.connect(config.DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO messages (chat_id, user_id, user_name, text, date)
            VALUES (?, ?, ?, ?, ?)
        """, (chat_id, user_id, user_name, text, date_ts))
        c.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat_id,))
        count = c.fetchone()[0]
        if count > config.MAX_HISTORY:
            to_remove = count - config.MAX_HISTORY
            logger.info(f"Слишком много сообщений ({count}) > {config.MAX_HISTORY}; удаляем {to_remove} старых.")
            c.execute("""
                DELETE FROM messages
                WHERE id IN (
                    SELECT id FROM messages
                    WHERE chat_id=?
                    ORDER BY id ASC
                    LIMIT ?
                )
            """, (chat_id, to_remove))
        conn.commit()

def fetch_last_messages(chat_id: int, limit=10):
    logger.info(f"fetch_last_messages(chat_id={chat_id}, limit={limit})")
    with sqlite3.connect(config.DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT user_name, text
            FROM messages
            WHERE chat_id=?
            ORDER BY id DESC
            LIMIT ?
        """, (chat_id, limit))
        rows = c.fetchall()
    rows.reverse()
    return rows

###########################
# OpenAI GPT (новый метод)
###########################

async def summarize_messages(messages) -> tuple[str, dict]:
    """
    Асинхронный вызов к OpenAI ChatCompletion с исправленным API.
    """
    logger.info(f"summarize_messages: входящих сообщений={len(messages)}")
    conversation_text = "\n".join(f"{user_name}: {text}" for user_name, text in messages)

    system_prompt = (
        "Ты — неформальный и ироничный рассказчик, который делает короткие, но ёмкие пересказы событий в чате. "
        "Стиль неофициальный, но понятный. Используй краткие предложения, добавляй легкий сарказм и юмор, если уместно. "
        "Не структурируй строго, но сохраняй логичный порядок событий. Формат должен быть живым и разговорным, "
        "примерно таким (но не копируй точь в точь, это скорее как идея):\n\n"
        "- [Событие 1, как будто рассказываешь другу]\n"
        "- [Событие 2, добавляешь легкий комментарий]\n"
        "- [Событие 3, если @Jack1008 спорил с @Glebady, отмечаешь, что он \"ожидаемо неправ\"]\n"
        "- [Событие 4, какие-то мысли или мечты участников]\n"
        "- [Дополнительный комментарий, что было, но не особо важно]\n\n"
        "Избегай формальной речи, не используй нумерацию, но передавай дух чата. Пиши как если бы пересказывал другу, "
        "который пропустил движ и хочет узнать, что случилось, но не в официальном тоне. Также можешь не упоминать короткие и незначительные детали."
    )

    usage_info = {}
    final_summary = "Упс, произошла ошибка при обращении к OpenAI."

    try:
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
            ],
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        final_summary = response.choices[0].message.content.strip()
        if response.usage:
            usage_info = response.usage
            logger.info(f"GPT usage: {usage_info}")
        else:
            logger.info("response.usage не обнаружен")
    except Exception as e:
        logger.error(f"OpenAI error: {e}")

    return final_summary, usage_info

###########################
# Антифлуд
###########################

def can_do_summary(chat_id: int) -> tuple[bool, str]:
    logger.info(f"can_do_summary({chat_id})?")
    last_summary_time, msg_count_at_summary = get_chat_info(chat_id)
    now_ts = int(time.time())
    dt_minutes = (now_ts - last_summary_time) / 60.0

    total_msgs = get_msg_count(chat_id)
    new_msgs = total_msgs - msg_count_at_summary

    logger.info(f"dt_minutes={dt_minutes:.1f}, new_msgs={new_msgs}")

    if dt_minutes < config.ANTI_FLOOD_MINUTES:
        need_wait = config.ANTI_FLOOD_MINUTES - dt_minutes
        return (False, f"Слишком часто просите пересказ. Подождите ещё ~{int(need_wait+1)} мин.")
    if new_msgs < config.MIN_NEW_MESSAGES:
        return (False, f"Нужно хотя бы {config.MIN_NEW_MESSAGES} новых сообщений. Сейчас только {new_msgs}.")
    return (True, "")

###########################
# Whitelist
###########################

async def chat_has_whitelisted_user(context: CallbackContext, chat_id: int) -> bool:
    if not config.WHITELIST_USERS:
        logger.info(f"chat_has_whitelisted_user({chat_id}): WHITELIST пуст => True")
        return True

    for wl_user in config.WHITELIST_USERS:
        try:
            if wl_user.isdigit():
                user_identifier = int(wl_user)
            else:
                user_identifier = wl_user.lstrip('@')
            member = await context.bot.get_chat_member(chat_id, user_identifier)
            logger.info(f"chat_has_whitelisted_user: get_chat_member => {user_identifier} status={member.status}")
            if member.status not in ["left", "kicked"]:
                return True
        except Exception as e:
            logger.info(f"chat_has_whitelisted_user: не нашли {wl_user} => {e}")

    logger.info(f"chat_has_whitelisted_user({chat_id}) => False (никто не найден)")
    return False

###########################
# Состояния Conversation
###########################
ST_CUSTOM_WAIT = 0

###########################
# on_chat_member_update
###########################

async def on_chat_member_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cm = update.chat_member
    if not cm:
        return

    chat_id = cm.chat.id
    user = cm.new_chat_member.user
    new_status = cm.new_chat_member.status

    logger.info(f"on_chat_member_update: chat={chat_id}, user={user.id} (@{user.username}), new_status={new_status}")

    if user.is_bot and user.id == context.bot.id:
        logger.info("Изменение статуса БОТа")
        if new_status in ["member", "administrator"]:
            try:
                has_wl = await chat_has_whitelisted_user(context, chat_id)
                if has_wl:
                    add_chat(chat_id)
                    logger.info(f"Бот добавлен в {chat_id}, зарегистрирован в БД.")
                else:
                    logger.info(f"Бот добавлен в {chat_id}, но нет whitelist => удаляем чат + leave_chat.")
                    remove_chat(chat_id)
                    try:
                        await context.bot.leave_chat(chat_id)
                    except Exception as e:
                        logger.info(f"Ошибка при leave_chat({chat_id}): {e}")
            except Exception as e:
                logger.info(f"Ошибка при проверке whitelist или add_chat: {e}")
        elif new_status in ["left", "kicked"]:
            logger.info(f"Бот покинул/кикнут {chat_id}, remove_chat.")
            remove_chat(chat_id)
    else:
        username = (user.username or "").lower()
        wl_list = [u.lstrip('@').lower() for u in config.WHITELIST_USERS]
        if username in wl_list:
            logger.info(f"Изменение статуса whitelist-пользователя: {username}")
            if chat_registered(chat_id):
                try:
                    bot_member = await context.bot.get_chat_member(chat_id, context.bot.id)
                    logger.info(f"Статус бота={bot_member.status} в чате {chat_id}")
                    if bot_member.status not in ["member", "administrator"]:
                        logger.info("Бот уже не член => remove_chat")
                        remove_chat(chat_id)
                    else:
                        has_wl = await chat_has_whitelisted_user(context, chat_id)
                        if has_wl:
                            logger.info("WL user changed, но всё ок, чат сохраняем.")
                        else:
                            logger.info("WL user ушёл => нет WL => бот уходит.")
                            remove_chat(chat_id)
                            try:
                                await context.bot.leave_chat(chat_id)
                            except Exception as e:
                                logger.info(f"Ошибка при leave_chat({chat_id}): {e}")
                except Exception as e:
                    logger.info(f"Ошибка при get_chat_member({chat_id}, бот) или проверке WL: {e}")

###########################
# store_text_message (ленивая регистрация)
###########################

async def store_text_message(update: Update, context: CallbackContext):
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user = update.effective_user
    text_preview = update.message.text[:30]
    logger.info(f"store_text_message: chat={chat_id}, user={(user.id if user else 0)}, text={text_preview}...")

    if not chat_registered(chat_id):
        logger.info(f"Чат {chat_id} не зарегистрирован => проверим состояние бота и whitelist.")
        try:
            bot_member = await context.bot.get_chat_member(chat_id, context.bot.id)
            logger.info(f"Статус бота в чате {chat_id}: {bot_member.status}")
            if bot_member.status in ["member", "administrator"]:
                has_wl = await chat_has_whitelisted_user(context, chat_id)
                if has_wl:
                    add_chat(chat_id)
                    logger.info(f"[store_text_message] Автоматически зарегистрировали чат {chat_id}, раз бот там + WL.")
                else:
                    logger.info(f"[store_text_message] bot_member={bot_member.status}, но нет WL => игнор/покинуть.")
            else:
                logger.info(f"[store_text_message] bot_member.status={bot_member.status}, не регистрируем.")
        except Exception as e:
            logger.info(f"Ошибка при get_chat_member({chat_id}, bot) или WL: {e}")

    if not chat_registered(chat_id):
        logger.info(f"Чат {chat_id} так и не зарегистрирован => не сохраняем сообщение.")
        return

    user_id = user.id if user else 0
    user_name = user.username if user else "Unknown"
    text = update.message.text
    date_ts = int(update.message.date.timestamp())
    save_message(chat_id, user_id, user_name, text, date_ts)

###########################
# Функции для выбора количества сообщений (с динамическим шагом)
###########################

def get_dynamic_step(value: int) -> int:
    """Возвращает шаг изменения в зависимости от текущего значения."""
    if value < 50:
        return 5
    elif value < 100:
        return 10
    else:
        return 50

async def summary_inline_custom(update: Update, context: CallbackContext):
    """
    После команды /summary сразу отображаем inline-клавиатуру из трёх кнопок:
    «−», число и «+».
    """
    chat_id = update.effective_chat.id
    user = update.effective_user
    logger.info(f"/summary вызван пользователем {user.id if user else 0} в чате {chat_id}")

    if not chat_registered(chat_id):
        await update.message.reply_text("Я не зарегистрирован в этом чате.")
        return ConversationHandler.END

    ok, reason = can_do_summary(chat_id)
    if not ok:
        logger.info(f"Антифлуд / мин. сообщений: {reason}")
        await update.message.reply_text(reason)
        return ConversationHandler.END

    context.user_data["summary_initiator_id"] = user.id if user else 0
    # Устанавливаем начальное значение
    context.user_data["custom_count"] = 50

    keyboard = [
        [
            InlineKeyboardButton("−", callback_data="custom_decrease"),
            InlineKeyboardButton(str(context.user_data["custom_count"]), callback_data="custom_confirm"),
            InlineKeyboardButton("+", callback_data="custom_increase")
        ]
    ]
    instruction = (
        "Используйте кнопки '−' и '+' для изменения количества сообщений.\n"
        "Нажмите на число, чтобы подтвердить выбор."
    )
    markup = InlineKeyboardMarkup(keyboard)
    sent = await update.message.reply_text(instruction, reply_markup=markup)
    logger.info(f"Сообщение с кнопками отправлено (message_id={sent.message_id}).")
    context.user_data["summary_message_id"] = sent.message_id
    return ST_CUSTOM_WAIT

async def custom_callback_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    user_id = query.from_user.id
    initiator_id = context.user_data.get("summary_initiator_id")
    if user_id != initiator_id:
        await query.answer("Эта кнопка не для вас!", show_alert=True)
        return

    data = query.data
    current_value = context.user_data.get("custom_count", 5)

    if data == "custom_increase":
        step = get_dynamic_step(current_value)
        current_value += step
        context.user_data["custom_count"] = current_value
    elif data == "custom_decrease":
        step = get_dynamic_step(current_value)
        current_value = max(1, current_value - step)
        context.user_data["custom_count"] = current_value
    elif data == "custom_confirm":
        chat_id = query.message.chat_id
        message_id = context.user_data["summary_message_id"]
        await do_summary_and_finish(chat_id, message_id, current_value, context)
        return ConversationHandler.END

    keyboard = [
        [
            InlineKeyboardButton("−", callback_data="custom_decrease"),
            InlineKeyboardButton(str(current_value), callback_data="custom_confirm"),
            InlineKeyboardButton("+", callback_data="custom_increase")
        ]
    ]
    instruction = (
        "Используйте кнопки '−' и '+' для изменения количества сообщений.\n"
        "Нажмите на число, чтобы подтвердить выбор."
    )
    markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(instruction, reply_markup=markup)

###########################
# do_summary_and_finish
###########################

async def do_summary_and_finish(chat_id: int, message_id: int, count: int, context: CallbackContext):
    logger.info(f"do_summary_and_finish(chat_id={chat_id}, count={count})")
    status_text = random.choice(config.STATUS_PHRASES)

    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=status_text
        )
        logger.info(f"Сообщение {message_id} в чате {chat_id} => изменено на статус '{status_text}'.")
    except Exception as e:
        logger.info(f"Ошибка при edit (статус): {e}")

    rows = fetch_last_messages(chat_id, limit=count)
    if not rows:
        final_text = "Не удалось найти сообщения для пересказа."
    else:
        summary, usage_info = await summarize_messages(rows)
        prompt_tokens = usage_info.prompt_tokens if usage_info else 0
        completion_tokens = usage_info.completion_tokens if usage_info else 0
        total_tokens = usage_info.total_tokens if usage_info else 0

        cost_input = (prompt_tokens / 1000) * config.GPT35_PRICE_PROMPT_PER_1K
        cost_output = (completion_tokens / 1000) * config.GPT35_PRICE_COMPLETION_PER_1K
        cost_total = cost_input + cost_output

        final_text = (
            f"{summary}\n\n"
            f"Этот пересказ стоил Лордису {total_tokens} токенов / "
            f"${cost_total:.5f}"
        )

    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=final_text
        )
        logger.info(f"Сообщение {message_id} => итоговый пересказ установлен (len={len(final_text)})")
    except Exception as e:
        logger.info(f"Ошибка при edit (итог): {e}")

    now_ts = int(time.time())
    total_msg_count = get_msg_count(chat_id)
    update_chat_info(chat_id, now_ts, total_msg_count)

###########################
# /summary Conversation
###########################

async def cmd_summary_entry(update: Update, context: CallbackContext):
    # Теперь сразу переходим к кастомному выбору
    return await summary_inline_custom(update, context)

###########################
# MAIN
###########################

def main():
    logger.info("Запуск main()...")

    if not config.BOT_TOKEN:
        logger.error("BOT_TOKEN не задан! (см. config.py/.env)")
        return
    if not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY не задан!")
        return

    init_db()

    app = ApplicationBuilder().token(config.BOT_TOKEN).build()
    logger.info("Телеграм-приложение создано (Application).")

    statuses = ["creator", "administrator", "member", "restricted", "left", "kicked"]
    chat_member_handler = ChatMemberHandler(on_chat_member_update, chat_member_types=statuses)
    app.add_handler(chat_member_handler)

    store_text_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, store_text_message)
    app.add_handler(store_text_handler)

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("summary", cmd_summary_entry)],
        states={
            ST_CUSTOM_WAIT: [CallbackQueryHandler(custom_callback_handler, pattern="^custom_")]
        },
        fallbacks=[],
        allow_reentry=True
    )
    app.add_handler(conv_handler)

    logger.info("Бот (polling) запускается...")
    app.run_polling(allowed_updates=[t.value for t in UpdateType], poll_interval=1.0)

if __name__ == "__main__":
    main()
