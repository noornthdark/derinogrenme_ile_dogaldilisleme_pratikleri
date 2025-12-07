"""
problem tanimi: kullanicilar yazili olarak soru soracak, gercek zamanli ve dogal selilde yanitlar alabilecek
    - akilli turizm rehberi
    - Türkiye özelinde: tarihi yerler, kültürel etkinlikler, yemekler, ulaşım ...
    - llama 3.2 3b parametreli modeli ile cevaplari streamlit uzerinden gercek zamanli olarak gorsellestirelim

model tanitimi: LLAMA (Large Language Model Meta AI) (llama 3.2 3B) 
    - acik kaynak: akademik ve ticari kullanimlar icin uygundur
    - verimli: daha az parametre ile ayni performansi sergiliyor
    - moduler: 1B, 3B, 8B, 70B paremetreye sahip modelleri var
    - lokal de calisabilir.
    - turkce için sorunlar cikabilir
    
Plan/Program

install libraries: freeze 

ollama indir ve llama kur
    - ollama: https://ollama.com/download
    - llama 3.2 3b: https://ollama.com/library/llama3.2:3b

import libraries
"""

# import libraries
from langchain.chat_models import ChatOllama # ollama llm arayuzu
from langchain.schema import SystemMessage, HumanMessage # chat mesaj siniflari
from langchain.memory import ConversationBufferMemory # konusma gecmisi icin basit bir hafiza

# llama model
llm = ChatOllama(model = "llama3.2:3b")

# hafiza ekleme, konusma gecmisi takip etme
memory = ConversationBufferMemory(return_messages=True) # return_messages = True -> mesajlar formatli doner

# welcome message
print("Akıllı Turizm Rehberine Hoş Geldiniz")
print("Size gezilecek yerler, tatil önerileri ve ulaşım bilgileri gibi konularda yardımcı olabilirim.")

# terminal uzerinden llama ile konusma
while True:

    user_input = input("Siz: ")

    if user_input.lower() == "quit":
        print("Program sonlandirildi")
        break

    # kullanicinin mesajlarini hafizaya kaydediyoruz
    memory.chat_memory.add_user_message(user_input)

    # model icin gerekli olan tum mesajlari olustur: sistem mesajı + memory + human mesajı
    messages = [
        SystemMessage(content = "Sen bir akıllı turizm rehberisin."
                      "Kullanıcılara Türkiye'de ki şehirler, tarihi yerler, yöresel yemekler, ulaşım ve tatil önerileri hakkında yardımcı ol.")
    ] + memory.load_memory_variables({})["history"] + [HumanMessage(content=user_input)]

    # modelden yanit alma
    response = llm(messages)

    # modelin cevabini hafizaya ekle
    memory.chat_memory.add_ai_message(response.content)

    print(f"Rehber ai: {response.content}")