import sqlite3 # SQLiter veritabani islemleri icin gerekli modul
import os

# veritabani dosyasinin yolunu olustur data/assistant.db
DB_PATH = os.path.join("data", "assistant.db")

# veritabani baslatan fonksiyon
def initialize_db():
    # eger data klasoru yoksa olustursun
    os.makedirs("data", exist_ok=True)

    # veritabanina baglan ve dosya yoksa olustur
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # eger notes tablosu yoksa olustur
    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS notes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,                   -- otomatik artan birincil anahtar
                        content TEXT NOT NULL,                                  -- not icerigi
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP          -- varsayilan olarak suanki zaman
                   )   
                """)
    
    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS calendar (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,  
                        event TEXT NOT NULL,                                    -- etkinlik aciklamasi bos olamaz
                        event_date TEXT NOT NULL                                -- etkinlik tarihi
                   )
        """)
    
    # degisiklikleri kayder 
    conn.commit()

    # baglantiyi kapat 
    conn.close()

# veritabanina yeni not ekleme islemi
def add_note(content):
    # veritabanina baglan
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # content i "notes" tablosuna ekle
    cursor.execute("INSERT INTO notes (content) VALUES (?)", (content,))

    # degisiklikleri kaydet
    conn.commit()

    # baglantiyi kapat
    conn.close()

# veritabanina yeni bir etkinlik ekleyen fonksiyon
def add_event(event, event_date):
    # veritabanina baglan
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # etkinlik ve tarih bilgilerini "calendar" tablosuna ekle
    cursor.execute("INSERT INTO calendar (event, event_date) VALUES (?, ?)", (event, event_date))

    # degisiklikleri kaydet
    conn.commit()

    conn.close() # baglantiyi kapat

# tum notlari veritabanindan sirali bir sekilde getiren fonksiyon
def get_notes():
    # veritabanina baglan
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # "notes" tablosundan icerik ve tarih bilgilerini zaman sirasina gore getir
    cursor.execute("SELECT content, created_at FROM notes ORDER BY created_at DESC")

    # sonuclari liste olarak alalim
    notes = cursor.fetchall()

    # baglantiyi kapat
    conn.close()

    return notes

# tum etkinlikler veritabanindan sirali sekilde getiren fonksiyon
def get_events():
    # veritabanina baglan
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # "calendar" tablosundan etkinlikleri tarihe gore siralayip getirelim
    cursor.execute("SELECT event, event_date FROM calendar ORDER BY event_date")

    # sonuclari al
    events = cursor.fetchall()

    # baglantiyi kapat
    conn.close()

    return events

if __name__ == "__main__":
    initialize_db()
    add_note("eve donerken su alamayi unutma")
    add_event("toplanti var", "15.12.2027")

    print(f"Notes: {get_notes()}")
    print(f"Events: {get_events()}")