"""
problem tanimi: LSTM ile metin uretme: verilen kelimelerden anlamli turkce cumleler olusturmasi
    - ben yarin ...

lstm: long short term memory

veri seti: chatgpt ile olusuturaln 100 adet gunluk hayat cumlesi

plan/program:

install libraries (pip), requirements.txt

import libraries
"""

# import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# egitim verisini chatgpt ile olustur
data = [
    "Bugün hava top oynamak için çok güzel.",
    "Kahvaltıda peynirli tost ve çay yedim.",
    "Sabah işe gitmek gerçekten zor geliyor.",
    "Arkadaşlarımla akşam sinemaya gideceğim.",
    "Telefonumun şarjı yine çok çabuk bitti.",
    "Bu sabah alarmı duymadım ve geç kaldım.",
    "Kitap okumak bana huzur veriyor.",
    "Akşam yemeğinde ne pişirsem bilmiyorum.",
    "Yarın için güzel planlarım var.",
    "Bugün çok verimli bir gün geçirdim.",
    "Sabah koşusu yapmayı seviyorum.",
    "Sürekli erteleme huyumdan kurtulmam lazım.",
    "Dışarıda yağmur yağıyor, evde kalmak en iyisi.",
    "Yeni bir diziyi izlemeye başladım.",
    "Kahve içmeden güne başlamak zor geliyor.",
    "İşten sonra biraz yürüyüş yapacağım.",
    "Yarın sabah erken kalkmam gerekiyor.",
    "Müzik dinlemek moralimi düzeltiyor.",
    "Bugün arkadaşım bana güzel bir haber verdi.",
    "Sıcak bir duş almak iyi geldi.",
    "Bütün gün bilgisayar başında çalıştım.",
    "Evde biraz temizlik yapmam lazım.",
    "Markete uğrayıp birkaç şey almam gerekiyor.",
    "Bugün kendimi biraz yorgun hissediyorum.",
    "Yeni tarifler denemek istiyorum.",
    "Kedim sabah beni uyandırdı.",
    "Bugün çok trafik vardı, geç kaldım.",
    "Biraz temiz hava almak istiyorum.",
    "Yarın için sunum hazırlamam lazım.",
    "Bugün hiç dışarı çıkasım yok.",
    "Kütüphanede çalışmak daha verimli oluyor.",
    "Kargo hala gelmedi, beklemekten sıkıldım.",
    "Kafamı dağıtmak için yürüyüşe çıktım.",
    "Yemek yaparken müzik dinlemek keyifli oluyor.",
    "Gün boyunca bir fincan kahve yetmedi.",
    "Yarınki toplantıya iyi hazırlanmalıyım.",
    "Bugün hava biraz serin ama güzel.",
    "Uyandığımda dışarısı kapkaranlıktı.",
    "Sınav haftası geldi, biraz stresliyim.",
    "Yeni kitapçıdan birkaç roman aldım.",
    "Bugün çalışmak yerine film izlemeyi tercih ettim.",
    "Sabah erkenden uyanmak bana zor geliyor.",
    "Ders çalışırken dikkatim çok çabuk dağılıyor.",
    "Bugün spor salonuna gitmek istemiyorum.",
    "Yeni aldığım ayakkabılar çok rahat.",
    "Pazar günleri tembellik yapmak hoşuma gidiyor.",
    "Dışarı çıkmadan önce hava durumuna bakıyorum.",
    "Hafta sonu piknik yapmayı planlıyoruz.",
    "Bugün kendime biraz zaman ayıracağım.",
    "Dışarıda güneş var ama içimde hep bir bulut.",
    "Sabah kahvemi içmeden güne başlayamıyorum.",
    "Bugün işe gitmek içimden gelmiyor.",
    "Akşam yemeğini dışarıda yemeyi düşünüyoruz.",
    "Dün gece çok geç yattım, uykusuzum.",
    "Kütüphanede sessizce ders çalışmak huzur veriyor.",
    "Hafta sonu için şehir dışına çıkacağım.",
    "Yeni telefonumun kamerası gerçekten çok iyi.",
    "Arkadaşımın doğum günü partisi çok eğlenceliydi.",
    "Bugün pazardan taze meyve aldım.",
    "Sürekli çalan bildirimler beni yoruyor.",
    "Evin camlarını silmem gerekiyor.",
    "Yeni aldığım kitapları bir solukta bitirdim.",
    "Sabahları güneş ışığıyla uyanmak çok güzel.",
    "Bu hafta işler çok yoğundu.",
    "Yolda yürürken eski bir arkadaşımla karşılaştım.",
    "Her gün en az on dakika meditasyon yapıyorum.",
    "Kahvaltıda zeytin, peynir ve yumurta vardı.",
    "Bugün çok güzel bir manzara gördüm.",
    "Kedim bütün gün koltukta uyudu.",
    "Yeni diziyi izlemeye başladım, çok sürükleyici.",
    "Bugün markette indirim vardı, biraz alışveriş yaptım.",
    "Çamaşırları kurutmak için balkona astım.",
    "Yarın erkenden uyanmam gerekiyor.",
    "Gün boyu bilgisayara bakmaktan gözüm ağrıdı.",
    "Telefonumu şarja takmayı unutmuşum.",
    "Sahilde yürüyüş yapmak bana çok iyi geliyor.",
    "Bu sabah kahvaltıyı dışarıda yaptım.",
    "Ders çalışmak yerine dizi izledim.",
    "Yeni tarif denemek için mutfağa girdim.",
    "Bugün kendime küçük bir ödül verdim.",
    "Hava biraz kapalı ama yine de dışarı çıkacağım.",
    "Yolda giderken yağmura yakalandım.",
    "Bu hafta biraz spor yapmam gerekiyor.",
    "Yarın sınavım var, biraz stresliyim.",
    "Dün gece çok garip bir rüya gördüm.",
    "Bugün kendimi çok enerjik hissediyorum.",
    "Uzun zamandır sinemaya gitmemiştim.",
    "Evde yalnız kalmak bazen iyi geliyor.",
    "Gün içinde su içmeyi sık sık unutuyorum.",
    "Kütüphanede ders çalışmak daha motive edici.",
    "Bugün öğle yemeğinde makarna yedim.",
    "Sabah dışarı çıkarken montumu unuttum.",
    "Yeni projeye başlamak beni heyecanlandırıyor.",
    "Her hafta sonu biraz kitap okumaya çalışıyorum.",
    "Bugün toplantı çok uzun sürdü.",
    "Kahve makinem yine bozuldu.",
    "Telefonumun ekranı çatladı, tamir ettirmeliyim.",
    "Hafta sonu ailemi ziyarete gideceğim.",
    "Evdeki saksı çiçeklerine su vermeyi unuttum.",
    "Bu sabah yürüyüş sırasında bir kedi peşimden geldi.",
    "Yarın hava nasıl olacak, merak ediyorum."
]

# -- Preprocessing --
# kelimeleri indexlere (sayilar) cevir (tokenizer)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1 # +1: padding icin yapiyoruz

print(f"total_words: {total_words}")

# n-gram dizileri olustur yani her cumleden kisa diziler olustur (embedding)
# 3-gram: kelimeleri indexlere (sayilar) cevir, ["kelimeleri indexlere (sayilar)", "indexlere (sayilar) cevir"]
input_sequences = []
for text in data:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i+1]
        input_sequences.append(n_gram_sequence)

print(f"input_sequences: \n{input_sequences}")
"""
[1, 9], [1, 9, 97], [1, 9, 97, 98], [1, 9, 97, 98, 7], [1, 9, 97, 98, 7, 2], [1, 9, 97, 98, 7, 2, 10],

"Bugün(1) hava(9) top(97) oynamak(98) için(7) çok(2) güzel.(10)"
"""
# padding: farkli uzunluktaki dizileri sabitle
max_sequence_length = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen = max_sequence_length, padding = "pre")

print(f"after padding input_sequences: \n{input_sequences}")
"""
[1, 9], [1, 9, 97], [1, 9, 97, 98],
[  0   0   0 ...   0   1   9]
 [  0   0   0 ...   1   9  97]
 [  0   0   0 ...   9  97  98]
"""
# girdi (X) ve hedef degiskenler (y) ayir
X = input_sequences[:, :-1] # n - 1 kelimeyi giris olarak sec
y = input_sequences[:, -1] # n inci kelimeyi tahmin et
"""
 [  0   0   0 ...   1   9  97]
 X = [  0   0   0 ...   1   9]
 y = [97]
"""
# hedef degiskene one hot encoding
y = tf.keras.utils.to_categorical(y, num_classes = total_words)
print(f"hedef degisken: {y}")
"""
[1,2,3] -> 
1 -> [1, 0, 0]
2 -> [0, 1, 0]
3 -> [0, 0, 1]
"""

# -- LSTM Training -- 
# lstm modeli tanimla
model = Sequential()
model.add(Embedding(total_words, 50, input_length = X.shape[1])) # embedding katmani
model.add(LSTM(100))
model.add(Dense(total_words, activation = "softmax")) # output cok sinifli classification"""
X = [bugün hava çok]
y = [güzel]

"""

# compile 
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

print(model.summary())

# egitimi baslat
# X = bagimsiz degiskenler
# y = bagimli degisken
# epoch verinin kac kere egitilecegi
# verbose = 1 ise egitim surecinin console da izlenmesi icin gerekli
model.fit(X, y, epochs = 100, verbose = 1)

# ornek uretim testi (metin uretimi)
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0] # tokenization
        token_list = pad_sequences([token_list], maxlen = max_sequence_length - 1, padding= "pre") # padding
        predicted_probs = model.predict(token_list, verbose = 0)
        predicted_index = np.argmax(predicted_probs, axis = -1)[0]
        predicted_word = tokenizer.index_word[predicted_index]
        seed_text = seed_text + " " + predicted_word 
    return seed_text

print(generate_text("Bugün", 5)) # Bugün çok trafik vardı geç kaldım

"""
(1)
seed_text = bu sabah
predicted_word = okula

(2)
seed_text = bu sabah okula
predicted_word = geç

(return)
seed_text = bu sabah okula geç
"""