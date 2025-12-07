"""
terminal uzerinden fastapi ile surekli sohbet (post atarak)
api endpoint: /chat
"""

import requests # http istekleri yapmak icin kullanilan kutuphane

# api adresi
API_URL = "http://127.0.0.1:8000/chat" # fast api sunucumuzun calistigi adres ve endpoint

# baslangicta kullanilan bilgileri al
name = input("Adiniz: ") 
age = int(input("Yasiniz: "))

print("\n Sohbet basladi. Cikmak icin quit yazin")

# bir dongu olustur, kullanicidan mesaj al ve sunucuya gonder
while True:

    user_msg = input(f"{name}: ") # kullanicidan mesaji al
    if user_msg.lower() == "quit":
        print("Program sonlandirildi.")
        break

    # API'ya gonderilecek veri paketi (json)
    payload = {
        "name": name,
        "age": age,
        "message": user_msg
    }

    try:
        # fastapi sunucusuna post istegi atalim, 30 saniye bekleyelim 
        res = requests.post(API_URL, json=payload, timeout=30)

        # eger istek basariliysa (200), yanit icinde responce kodu yazdirilir
        if res.status_code == 200:
            print(f"Doktor Asistanı: {res.json()["response"]}")
        else:
            print("hata", res.status_code, res.text)
    except requests.exceptions.RequestException as e:
        print("Baglanti hatasi: ", e)