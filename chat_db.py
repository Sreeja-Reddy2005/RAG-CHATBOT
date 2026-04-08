import sqlite3

DB_NAME = "chatbot.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def create_tables():
    conn = get_connection()
    c = conn.cursor()

  
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

   
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT
    )
    """)

 
    c.execute("""
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        role TEXT,
        content TEXT
    )
    """)

   
    c.execute("""
    CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        file_path TEXT
    )
    """)

   
    c.execute("""
    CREATE TABLE IF NOT EXISTS images(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        image_base64 TEXT
    )
    """)

    conn.commit()
    conn.close()



def create_conversation(user_id, title):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
        (user_id, title)
    )

    conn.commit()
    cid = c.lastrowid
    conn.close()
    return cid


def get_conversations(user_id):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        "SELECT id, title FROM conversations WHERE user_id=?",
        (user_id,)
    )

    data = c.fetchall()
    conn.close()
    return data



def save_message(cid, role, content):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (cid, role, content)
    )

    conn.commit()
    conn.close()


def get_messages(cid):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        "SELECT role, content FROM messages WHERE conversation_id=?",
        (cid,)
    )

    rows = c.fetchall()
    conn.close()

    return [{"role": r[0], "content": r[1]} for r in rows]



def save_document(cid, file_path):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        "INSERT INTO documents (conversation_id, file_path) VALUES (?, ?)",
        (cid, file_path)
    )

    conn.commit()
    conn.close()


def load_document_path(cid):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        """
        SELECT file_path 
        FROM documents 
        WHERE conversation_id=? 
        ORDER BY id DESC 
        LIMIT 1
        """,
        (cid,)
    )

    row = c.fetchone()
    conn.close()

    return row[0] if row else None


def save_image(cid, image_base64):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        "INSERT INTO images (conversation_id, image_base64) VALUES (?, ?)",
        (cid, image_base64)
    )

    conn.commit()
    conn.close()


def load_image(cid):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        """
        SELECT image_base64 
        FROM images 
        WHERE conversation_id=? 
        ORDER BY id DESC 
        LIMIT 1
        """,
        (cid,)
    )

    row = c.fetchone()
    conn.close()

    return row[0] if row else None
