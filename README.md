#  Yapay Zeka ile Çiçek Sınıflandırma (Transfer Learning)

Sivas Cumhuriyet Üniversitesi Şarkışla Uygulamalı Bilimler Yüksekokulu Derin Öğrenme dersi kapsamında geliştirilmiş bir derin öğrenme (Deep Learning) görüntü sınıflandırma modelidir[cite: 18]. Projede, TensorFlow'un `flower_photos` veri seti kullanılarak beş farklı çiçek türü (papatya, karahindiba, gül, ayçiçeği, lale) sınıflandırılmaktadır[cite: 18].

## Algoritma Seçimi ve Mimari
Geliştirme sürecinde klasik CNN, VGG16, ResNet50 ve DenseNet121 gibi toplam 10 farklı algoritma ve mimari; doğruluk potansiyeli, eğitim maliyeti ve tahmin hızı açısından karşılaştırılmıştır[cite: 18]. 
* Yüksek doğruluk oranı, hafif model boyutu ve Gradio web arayüzünde hızlı tahmin üretmesi sebebiyle önceden ImageNet üzerinde eğitilmiş **MobileNetV2** mimarisi en uygun model olarak seçilmiştir[cite: 18]. 
* Temel katmanlar dondurularak, çıkışa beş sınıflı yoğun (Dense) katman eklenmiş ve Adam optimizasyonu kullanılmıştır[cite: 18].

## Performans Analizi ve Metrikler
Model 10 epoch boyunca eğitilmiş ve test verileri üzerinde ~%91 doğruluk (accuracy) seviyesine ulaşmıştır[cite: 18].
* **En Yüksek Başarı:** Dandelion (karahindiba) sınıfında 0.96 F1-Score değeri elde edilmiştir[cite: 18].
* **Karışıklık (Confusion) Analizi:** Roses (gül) ve tulips (lale) sınıfları görsel ve yaprak yapısı benzerliklerinden dolayı matris üzerinde en çok karışan türler olmuştur[cite: 18].

## İnteraktif Web Arayüzü (Gradio)
Proje, kullanıcı dostu bir web paneli üzerinden test edilebilir şekilde Gradio kütüphanesiyle yayına alınmıştır[cite: 18]. Sistem, yüklenen görseli 224x224 boyutuna getirip softmax ile olasılıkları hesaplayarak en yüksek üç sınıfı ekranda gösterir[cite: 18].

## Gelecek Geliştirmeler (Future Work)
* Veri artırma (data augmentation) tekniklerinin eklenmesi[cite: 18].
* MobileNetV2 son katmanlarında fine-tuning uygulanması[cite: 18].
* Modelin TensorFlow Lite formatına dönüştürülerek mobil uygulamaya taşınması[cite: 18].