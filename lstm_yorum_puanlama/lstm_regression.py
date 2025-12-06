"""
Problem tanimi: yorumlardan -> puan tahmini (1-5), regresyon problemi 
    - cok iyiydi cok memnun kaldim -> 4.5
    - berbatti, bir daha gelmem -> 1.2

Veri seti: yelp dataset, hugging face, (restoran, doktor, otel, araba yikama ...)
    - text: yorum metni
    - label: 0-4 arasinda ama biz bunu 1ile 5 e cekelim.
    - https://huggingface.co/datasets/Yelp/yelp_review_full

LSTM: bir yorumu bastan sona okur, sonrasinda yorumun genel anlamina karsilik gelen yildiz puanini cikarir

install libraries: freeze requirements.txt

Plan/program

import libraries
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle # tokenizer'i diske kaydetmek icin kullanicaz

from sklearn.model_selection import train_test_split # veriyi egitim ve test olmak uzere 2 ye ayir
from sklearn.preprocessing import MinMaxScaler # normalization

from tensorflow.keras.preprocessing.text import Tokenizer # tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences # padding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense 
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError # 5 -> 4 = 1, 4 -> 5 = 1

# load yelp dataset
# hugging face den yelp veri setini yukleme
splits = {"train": "yelp_review_full/train-00000-of-00001.parquet"}
train_path = "hf://datasets/Yelp/yelp_review_full/" + splits["train"]

# parquet formatindan veriyi pandas ile oku
df = pd.read_parquet(train_path)
print(df.head())

# etiketleri 0-4 araligindan 1-5 araligina donustur
df["label"] = df["label"] + 1

# data preprocessing
texts = df["text"].values # yorum metinleri
labels = df["label"].values # puanlar 1-5 arasinda

# tokenizer: metni sayiya cevir
# num_words: en cok gecen ilk 10000 kelime
# OOV: bilinmeyen kelimeleri bu etiketle goster
tokenizer = Tokenizer(num_words = 10000, oov_token = "<OOV>")

# metni sayilara donustur
tokenizer.fit_on_texts(texts)

# tokenizeri diske kaydet
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# yorumlari dizi haline getir
sequences = tokenizer.texts_to_sequences(texts)

# tum dizileri sabit uunluga getir yani padding uygula (kisa olnlari 0 ile doldur)
padded_sequences = pad_sequences(sequences, maxlen = 100, padding = "post", truncating = "post")

# etiketler 1 ile 5 arasinda, normalization ile 0 iel 1 arasina alalim, cunku regresyon problemlerinde daha stabil bir ogrenme sagliyor
scaler = MinMaxScaler() # 1-5 - 1 = 0-4 sonra /4 = 0-1
labels_scaled = scaler.fit_transform(labels.reshape(-1, 1))

# egitim ve test verisini ayir
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels_scaled, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_train: {X_train[:2]}")
print(f"y_train shape: {y_train.shape}")
print(f"y_train: {y_train[:2]}")

# LSTM tabanli regresyon modeli
model = Sequential()

# embedding katmani: kelime indekslerini vektor uzayina donusturur
# input_dim: 10000 -> kelime sayisi
# output_dim: 128 -> her bir kelime 128 boyutlu vektorle temsil edilecek
# input_length: 100 -> sabit dizi uzunlugu yani her bir metnimizin uzunlugu 
model.add(Embedding(input_dim = 10000, output_dim = 128, input_length = 100))

# LSTM katmani: sirali veride baglami ogrenecek olan katman
model.add(LSTM(128)) # 128: lstm de bulunan hucre sayisi yani daha fazla ogrenme kapasitesi

# tam bagli (dense) layer
model.add(Dense(64, activation = "relu"))

# output layer
model.add(Dense(1, activation = "linear")) # relu, tanh, sigmoid, softmax, linear

# model compile and training
model.compile(
    optimizer = "adam", # adaptif ogrenme algoritmasi
    loss = MeanSquaredError(), # mean squared error: regresyon icin uygun bir loss fonksiyonu
    metrics = [MeanAbsoluteError()] # modelin hata ortalamasi
)

history = model.fit(
    X_train, y_train,
    epochs = 3, # toplam egitim dongusu
    batch_size = 64, # her adimda islenecek ornek sayisi
    validation_split = 0.2 # egitim verisinin %20 si validasyon icin ayrilir 
    )


# egitim kayip grafigini gorsellestir ve modeli kaydet
plt.plot(history.history["loss"], label= "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.title("Egitim sureci MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss MSE")
plt.show()

# modeli kaydet
model.save("regression_lstm_yelp.h5")