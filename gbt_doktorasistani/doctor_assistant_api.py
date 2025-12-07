"""
Fast API ile GPT Doktor asistani
her kullanici icin ayri bir memory tutalim
"""

import os
from typing import Dict

from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# ortam degiskenleri
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# fast api uygulamasini baslat
app = FastAPI(title = "Doktor Asistanı API")

# LLM Yapilandirma
llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    temperature = 0.7, 
    openai_api_key = api_key
)

# Memory yapilandirmasi
user_memories: Dict[str, ConversationBufferMemory] = {}

# istek ve yanit semalari
class ChatRequest(BaseModel): # chat mesaji input
    name: str
    age: int
    message: str

class ChatResponse(BaseModel): # chat cevabi output
    response: str

# sohbet endpoint
@app.post("/chat", response_model = ChatResponse)
async def chat_with_doctor(request: ChatRequest):
    try:
        # hafiza varsa hafizayi getir, eger yoksa da hafizayi yarat
        if request.name not in user_memories:
            user_memories[request.name] = ConversationBufferMemory(return_messages=True)
    
        memory = user_memories[request.name]

        # intro mesajini olustur
        if len(memory.chat_memory.messages) == 0:
            intro = (
                f"sen bir doktor asistanısın. Hasta: {request.name}, {request.age} yaşında. "
                "Sağlık sorunları hakkında konuşmak istiyor. "
                "Yasşına uygun, dikkatli ve nazik tavsiyeler ver. "
                "Kullanıcıya ismiyle hitap et."
            )
            memory.chat_memory.add_user_message(intro)
        
        # llm ile memory birlestir, chain olustur
        conversation = ConversationChain(llm = llm, memory = memory, verbose = False)
        reply = conversation.predict(input = request.message)

        # hafizayi terminale yazdir
        print(f"\n Memory: ")
        for idx, m in enumerate(memory.chat_memory.messages, start = 1):
            print(f"{idx:02d}. {m.type.upper()}: {m.content}")
        print("------------------------------------------------------------")

        return ChatResponse(response = reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# swagger: http://127.0.0.1:8000/docs


