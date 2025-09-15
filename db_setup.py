import sqlite3

conn = sqlite3.connect("inspection.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    qr_code TEXT NOT NULL,
    details TEXT,
    inspector_name TEXT,
    inspection_photo BLOB,
    gps_location TEXT,
    timestamp TEXT,
    ml_prediction TEXT  -- New column
)
""")

conn.commit()
conn.close()

print("Database & table created successfully.")