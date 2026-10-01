import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

tf.keras.backend.clear_session()

print("1. VERİ SETİ İNDİRİLİYOR...")
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('cicek_deposu', origin=dataset_url, extract=True)

dogru_klasor = ""
for root, dirs, files in os.walk(os.path.dirname(data_dir)):
    if 'daisy' in dirs and 'roses' in dirs:
        dogru_klasor = root
        break

print("2. VERİLER OKUNUYOR...")
goruntu_boyutu = (224, 224)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dogru_klasor, validation_split=0.2, subset="training", seed=123, image_size=goruntu_boyutu, batch_size=32)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    dogru_klasor, validation_split=0.2, subset="validation", seed=123, image_size=goruntu_boyutu, batch_size=32)

sinif_isimleri = train_dataset.class_names

print("3. BEYİN HAZIRLANIYOR (MobileNetV2)...")
temel_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
temel_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    temel_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(sinif_isimleri))
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("4. EĞİTİM BAŞLIYOR...")
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

print("5. MODEL KAYDEDİLİYOR...")
model.save_weights('cicek_beyni.weights.h5')
print("Model başarıyla kaydedildi!")