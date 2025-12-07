import streamlit as st

from langchain.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

# streaming callbacks
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # terminale yazmak
from langchain.callbacks.base import BaseCallbackHandler # streamlit ile calismak icin ozel handler
from typing import Any

# streamlit icin ozel streaming callback tanimi
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder # streamlit icindeki mesaj kutumuz
        self.final_text = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.final_text += token # tokenlari birlestir
        self.placeholder.markdown(self.final_text + " ") # canli olarak yaz

# baslik ve aciklamalar
st.set_page_config(page_title = "Akıllı Turist Rehberi (Canlı)", page_icon = "🌍")
st.title("🌍 Akıllı Turist Rehberi (Streaming Modu)")
st.markdown("Türkiye'nin dört bir yanındaki turistik yerler hakkında bilgi almak için sorular sorabilirsiniz.")

# session state (streamlit de kullanici gecmisini tutmak icin)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory( return_messages=True) # mesaj gecmisi

# mesaj kutusu: kullanicidan gelen mesaj
user_input = st.chat_input("Bir şehir, mekan, yemek ya da aktivite sorabilirsiniz...")

# sohbet gecmisini arayuzde goster
# tum mesajlari sirasiyla gezip ekrana bastiralim
for msg in st.session_state.memory.chat_memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("🧑‍💼 Kullanıcı"):
            st.markdown(msg.content)
    else: # ai ise 
        with st.chat_message("🧭 Akıllı Rehber"):
            st.markdown(msg.content)

if user_input:
    # yeni gelen kullanici mesajini ilk olarak memory e ekliyoruz
    st.session_state.memory.chat_memory.add_user_message(user_input)
    with st.chat_message("🧑‍💼 Kullanıcı"):
        st.markdown(user_input)

    with st.chat_message("🧭 Akıllı Rehber"):

        response_placeholder = st.empty() # streamlitte geciic mesaj kutusu
        stream_handler = StreamHandler(response_placeholder) 

        llm = ChatOllama(model = "llama3.2:3b", streaming = True, callbacks = [stream_handler])

        # tum konusmayi modele verecek sekilde mesajlari olusturalim: sistem mesaji + memory + human message
        messages = [
            SystemMessage(content = "Sen akıllı turizm ve turist rehberisin. "
                        "kullanıcılara Türkiye'de ki şehirler, tarihi yerler, yöresel yemekler, ulaşım ve tatil önerileri hakkında güzel bilgiler ver. ")                   
        ] + st.session_state.memory.load_memory_variables({})["history"] + [HumanMessage(content = user_input)]

        # modelden yanit al
        response = llm(messages)

        # yaniti hafizaya kaydet
        st.session_state.memory.chat_memory.add_ai_message(response.content)