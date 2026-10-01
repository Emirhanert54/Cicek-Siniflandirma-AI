import gradio as gr
import tensorflow as tf
import numpy as np

print("1. Sistem iskeleti kuruluyor...")
sinif_isimleri = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

temel_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
temel_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    temel_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5)
])

print("2. Eğitilmiş beyin (ağırlıklar) yükleniyor...")
# train.py'den çıkan ağırlıkları yüklüyoruz
model.load_weights('cicek_beyni.weights.h5') 

def cicek_tahmin_et(img):
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    tahminler = model.predict(img_array)
    skorlar = tf.nn.softmax(tahminler[0])

    sonuclar = {}
    for i, sinif in enumerate(sinif_isimleri):
        sonuclar[sinif.capitalize()] = float(skorlar[i])
    return sonuclar

print("3. Web sitesi ayağa kaldırılıyor...")
arayuz = gr.Interface(
    fn=cicek_tahmin_et,
    inputs=gr.Image(type="pil", label="Fotoğraf Yükleyin"),
    outputs=gr.Label(num_top_classes=3, label="Yapay Zekanın Tahmini"),
    title="🌸 Yapay Zeka Çiçek Tanıma Sistemi",
    description="MobileNetV2 altyapısı kullanılarak 5 farklı çiçek türünü tanımak üzere eğitilmiştir. Yarım saniyede sonuç verir."
)

arayuz.launch(debug=True, share=True)