"""
web uzerinde calisan chatbot ekrani gelistirme 
streamlit framework
"""

import streamlit as st # streamlit ile web arayuzu olusturma kutuphanesi
from langchain.chat_models import ChatOllama # ollama uzerinden llama cagirmak icin
from langchain.schema import SystemMessage, HumanMessage # sohbet mesajlari
from langchain.memory import ConversationBufferMemory # hafiza yonetimi

# baslik ve aciklamalar
st.set_page_config(page_title = "Akıllı Turist Rehberi", page_icon = "🌍")
st.title("🌍 Akıllı Turist Rehberi")
st.markdown("Türkiye'nin dört bir yanındaki turistik yerler hakkında bilgi almak için sorular sorabilirsiniz.")

# session state (streamlit de kullanici gecmisini tutmak icin)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory( return_messages=True) # mesaj gecmisi

# ollama ile llama 3.2 3B parametreli modeli yukleyelim
llm = ChatOllama(model = "llama3.2:3b")

# mesaj kutusu: kullanicidan gelen mesaj
user_input = st.chat_input("Bir şehir, mekan, yemek ya da aktivite sorabilirsiniz...")

if user_input:
    # yeni gelen kullanici mesajini ilk olarak memory e ekliyoruz
    st.session_state.memory.chat_memory.add_user_message(user_input)

    # tum konusmayi modele verecek sekilde mesajlari olusturalim: sistem mesaji + memory + human message
    messages = [
        SystemMessage(content = "Sen akıllı turizm ve turist rehberisin. "
                      "kullanıcılara Türkiye'de ki şehirler, tarihi yerler, yöresel yemekler, ulaşım ve tatil önerileri hakkında güzel bilgiler ver. ")                   
    ] + st.session_state.memory.load_memory_variables({})["history"] + [HumanMessage(content = user_input)]

    # modelden yanit al
    response = llm(messages)

    # yaniti hafizaya kaydet
    st.session_state.memory.chat_memory.add_ai_message(response.content)

# sohbet gecmisini arayuzde goster
# tum mesajlari sirasiyla gezip ekrana bastiralim
for msg in st.session_state.memory.chat_memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("🧑‍💼 Kullanıcı"):
            st.markdown(msg.content)
    else: # ai ise 
        with st.chat_message("🧭 Akıllı Rehber"):
            st.markdown(msg.content)