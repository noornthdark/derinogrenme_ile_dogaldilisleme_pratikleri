import os # ortam degiskenleri ve dosya yolu
import requests # http istekleri yapmak icin
from dotenv import load_dotenv # .env dosyasindan ortam degiskenlerini yuklemek icin

# .env dosyasini yukle
load_dotenv()

# .env dosyasindaki GEMINI_API_KEY degiskenini alalim
api_key = os.getenv("GEMINI_API_KEY")

# eger api anahtari yoksa kullaniciya hata gonder
if not api_key:
    raise ValueError("GEMINI_API_KEY .env dosyasinda tanimli degil")

# gemini 2.0 flash modeline ait api url'i (google ai tarafindan aldik)
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# api cagrisi icin gerekli http basliklari
headers = {
    "Content-Type": "application/json", # JSON formatinda veri gonderecegimizi belirtiyoruz
    "X-Goog-Api-Key": api_key # yetkilendirme icin api anahtari
}

def get_gemini_response(prompt: str) -> str: # gemini api'sine prompt gonderip yanit return eden fonksiyon
    # api'ye gonderilecek json yapisi
    payload = {
        "contents":[
            {
                "parts":[
                    {"text":prompt} # kullanicidan gelen mesaji iceren bolum
                ]
            }
        ]
    }

    # gemini api ye http post istegi gonder
    response = requests.post(url, headers = headers, json = payload)

    # istek basariliysa (http 200) 
    if response.status_code == 200:
        try: 
            result = response.json() # json formatindaki yaniti sozluge cevirizz
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            # eger json yapisi beklenildigi gibi degilse hata dondurur
            return f"Yanit hatasi: {e}"
    else:
        return f"api hatasi {response.status_code}: {response.text}"
    
# kullanici mesajina gore niyet siniflandirmasi yapan bir fonksiyon
def detect_intent(message):
    # gemini icin ozel bir gorev promtu olustur: mesajin hangi kategoriye ait oldugunu tespit etsin
    prompt = f"""
                Kullanıcının aşağıdaki cümlesini sınıflandır:

                Etiketlerden sadece birini döndür:
                - not_ozet (eğer notları özetlemesini istiyorsa)
                - etkinlik_ozet (eger etkinlikleri görmek yada özet istiyorsa)
                - normal (diğer her şey)

                Cümle: "{message}"
                Yalnızca etiket döndür: (örnek: not_ozet)
            """
    # promtu gemini a gonder ve cevap al
    response = get_gemini_response(prompt)
    return response.strip().lower()

if __name__ == "__main__":
    user_input = input("Kullanici Sorusu: ") # kullanicdan terminal uzerinde girdi almak icin
    yanit = get_gemini_response(user_input) # gemini den alinan yanit return edilir
    print(f"Akilli Asistan yaniti: {yanit}")