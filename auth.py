import sqlite3
import hashlib

def get_db():
    return sqlite3.connect("chatbot.db")

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def register(u,p):
    conn = get_db()
    c = conn.cursor()

    try:
        c.execute("INSERT INTO users(username,password) VALUES(?,?)",
                  (u,hash_password(p)))
        conn.commit()
        return True
    except:
        return False

def login(u,p):
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT id,password FROM users WHERE username=?", (u,))
    user = c.fetchone()

    if user and user[1] == hash_password(p):
        class User:
            def __init__(self,id):
                self.id=id
        return User(user[0])

    return None