import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "students.db")

def connect_database():
    """Open a SQLite connection and return rows as mapping-like objects."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection

def init_database():
    """Create the students table automatically on first launch."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with connect_database() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                branch TEXT NOT NULL,
                semester INTEGER NOT NULL CHECK (semester BETWEEN 1 AND 8),
                email TEXT,
                mobile TEXT,
                math_marks REAL NOT NULL CHECK (math_marks BETWEEN 0 AND 100),
                python_marks REAL NOT NULL CHECK (python_marks BETWEEN 0 AND 100),
                cn_marks REAL NOT NULL CHECK (cn_marks BETWEEN 0 AND 100),
                wad_marks REAL NOT NULL CHECK (wad_marks BETWEEN 0 AND 100),
                average_marks REAL NOT NULL,
                attendance REAL NOT NULL CHECK (attendance BETWEEN 0 AND 100),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
