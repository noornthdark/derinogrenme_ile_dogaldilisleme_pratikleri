"""
Problem Tanimi: Gemini ile Akıllı Asistan Projesi: notlar, gorevler ve etkinlikler icin akıllı asistan
    - Amac: google gemini api tabanli yapay zeka kullanan bir akıllı asistan gelistirme
    - kullanıcın dogal dilde verdiği komutları anlar (sohbet botu)
    - Kural tabanli olarak notlar ve etkinlikler olusturalim
    - akıllı asistanımız notlara ve etkinlikler eerişim sağlayarak bie öetleme, bilgi çıkarma takvim oluşturma gibi özellikler sunar

Model Tanitimi: Google DeepMind Gemini
    - Bu projede gemini-2.0-flash modelini kullanicaz

API Tanimlama: https://ai.google.dev/gemini-api/docs?hl=tr

plan/program:
    - assistant: gemini chatbot olusturulur
    - database: sqlite database olusturalim, notlar ve etkinlikleri depolamak lazım
    - main: bilesenleri bir araya getirir

install libraries
"""

# assistant.py dosyasindan gemini api yanitini alan ve kullanicinin niyetini belirleyen fonksiyonlari icerir
from assistant import get_gemini_response, detect_intent

# database.py dosyasindan veritabani islemleri gerceklestiren yardimci fonksiyonlarimiz
from database import initialize_db, add_event, add_note, get_notes, get_events

# veritabanini baslat
initialize_db()

# karsilama mesaji
print("Akıllı Asistana Hoşgeldiniz")
print("Komutlar: not ekle | etkinlik ekle | notları göster | etkinlikleri göster | sohbet et | çıkış")

# kullanicidan surekli komut almak icin sonsuz dongu baslat
while True:

    komut = input("Komut girin: ").strip().lower() # komutu al, bosluklari kirp, kucuk harfe cevir

    if komut == "not ekle":
        content = input("Not içeriği nedir? ")# kullancııdan not içeriği al
        add_note(content)
        print("Not başarıyla kaydedildi. ")
    elif komut == "etkinlik ekle":
        event = input("Etkinlik açıklaması? ")
        date = input("Etkinlik tarihi? ")
        add_event(event, date)
        print("Etkinlik eklendi.")
    elif komut == "notları göster":
        notes = get_notes() # veri tabanindan tum notlari al
        if notes:
            print("Kaydedilmiş notlar: ")
            for content, created_at in notes: # her notu ve tarihi yazdır
                print(f"- [{created_at}] {content}")
        else:
            print("Henüz hiç bir not eklenmedi.")
    elif komut == "etkinlikleri göster":
        events = get_events()
        if events:
            print("Etkinlikler: ")
            for event, event_date in events:
                print(f"- {event_date}: {event}")
        else:
            print("Henuz etkinlik girilmemiş")
    elif komut == "sohbet et":
        message = input("Kullanici: ").strip() # kullanıcıdan serbest metin al
        intent = detect_intent(message) # kullanicinin niyetini (not ozeti mi, etkinlik ozeti mi, havadan sudan sohbet mi) anlama 
        
        if intent == "not_ozet":
            notes = get_notes() # notlari veri tabanindan al
            if not notes:
                print("Henuz ozetlenecek not blunmuyor")
                continue
            
            all_notes_text = "\n".join([f"- {note[0]}" for note in notes]) # tum notlari birlestir
            prompt = f"Aşağıda bulunan notları özetler misin \n\n {all_notes_text}" # gemini dan ozet ise 
            summary = get_gemini_response(prompt) # gemini dan ozet iste

            print("Not özeti: \n")
            print(summary)
        elif intent == "etkinlik_ozet":
            events = get_events() # etkinlikleri veri tabanindan al
            if not events:
                print("Henuz ozetlenecek etkinlik bulunamadi")
                continue
        
            all_events_text = "\n".join([f"- {e[1]}: {e[0]}" for e in events]) # tum etkinlikerli listele
            prompt = f"Aşağıdaki takvim etkinliklerini kullanıcı isteğine göre özetler misin\n\n{all_events_text}\n\n kullanıcı isteği: {message}"
            summary = get_gemini_response(prompt) # geminidan ozet ister
            print("Etkinlik özeti: \n")
            print(summary) 
        else:
            reply = get_gemini_response(message)
            print(f"Akıllı Asistan: {reply}")
    elif komut == "çıkış":
        break
    else:
        print("hatalı komut girdiniz.")








