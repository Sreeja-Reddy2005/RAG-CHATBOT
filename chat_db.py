import sqlite3

DB_NAME = "chatbot.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

   
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

 
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT
    )
    """)

   
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        role TEXT,
        content TEXT,
        image TEXT
    )
    """)

 
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        summary TEXT
    )
    """)


    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        content BLOB
    )
    """)

    conn.commit()
    conn.close()


def create_conversation(user_id, title):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
        (user_id, title)
    )

    conn.commit()
    cid = cursor.lastrowid
    conn.close()

    print(" Created conversation:", cid)
    return cid


def save_message(conversation_id, role, content, image=None):
    if conversation_id is None:
        print(" conversation_id is None")
        return

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, image) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, image)
    )

    conn.commit()
    conn.close()

def save_summary(conversation_id, summary):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO summaries (conversation_id, summary) VALUES (?, ?)",
        (conversation_id, summary)
    )

    conn.commit()
    conn.close()

def save_document(conversation_id, file_bytes):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO documents (conversation_id, content) VALUES (?, ?)",
        (conversation_id, file_bytes)
    )

    conn.commit()
    conn.close()

def load_document(conversation_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT content FROM documents WHERE conversation_id=? ORDER BY id DESC LIMIT 1",
        (conversation_id,)
    )

    row = cursor.fetchone()
    conn.close()

    return row[0] if row else None


def get_messages(conversation_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT role, content, image FROM messages WHERE conversation_id=?",
        (conversation_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    return [{"role": r[0], "content": r[1], "image": r[2]} for r in rows]


def get_conversations(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, title FROM conversations WHERE user_id=?",
        (user_id,)
    )

    data = cursor.fetchall()
    conn.close()

    return data