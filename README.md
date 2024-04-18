# Bau_Final_Projesi

Obezite, dünya genelinde milyonlarca insanı etkileyen ve hastalık, ölüm ve sağlık hizmetleri maliyetleri açısından önemli sonuçları olan acil bir küresel sağlık sorunudur. Obezitenin yaygınlığı 1975'ten bu yana üç katına çıkmış, şu anda küresel nüfusun yaklaşık %30'unu etkilemektedir. Bu artan trend, fazla kilo ile ilişkili çok yönlü risklerin acilen ele alınmasının gerekliliğini vurgulamaktadır. Obezite, diyabet, kalp hastalıkları, osteoartrit, uyku apnesi, felç ve yüksek kan basıncı gibi çeşitli sağlık komplikasyonlarının önde gelen nedenlerindendir ve yaşam beklentisini önemli ölçüde azaltır ve ölüm oranlarını artırır. Obezite riskinin etkili bir şekilde tahmin edilmesi, hedeflenmiş müdahalelerin uygulanması ve halk sağlığının teşvik edilmesi için hayati önem taşır.

STEP 1: VERİ ANALİZİ
1.Genel Bakış
-Veri setini yükledik ve temel bilgileri inceledik.

-Veri kümesindeki değişkenler ve bunların açıklamaları araştırıldı.

2.⁠ ⁠Sayısal ve Kategorik Değişkenlerin Belirlenmesi

-Kategorik, sayısal ve kategorik ancak temel değişkenleri tanımlamak için bir fonksiyon oluşturuldu.

-Veri setimizdeki kategorik, sayısal ve kategorik ancak temel değişkenleri tanımlamak için bu işlevi kullandık.

3.Kategorik Değişkenlerin Analizi

-Kategorik değişkenlerin frekans ve oranlarının incelenmesi.

-Kategorik değişkenlerin dağılımlarını grafiklerle görselleştirdim.

4.⁠ ⁠Sayısal Değişkenlerin Analizi

-Temel istatistikler ve sayısal değişkenlerin dağılımları incelendi.

-Sayısal değişkenlerin dağılımlarını grafiklerle görselleştirdim.

5.Hedef Değişken Analizi

-Kategorik ve sayısal değişkenlere dayalı olarak hedef değişkenin (NObeyesdad) ortalamaları araştırıldı.

6.⁠ ⁠Aykırı Değer Analizi

-Sayısal değişkenlerde tespit edilen aykırı değerler.

7.Eksik Veri Analizi

-Veri setindeki eksik gözlemler kontrol edildi.

8.⁠ ⁠Korelasyon Analizi ve Görselleştirme

-Sayısal değişkenler arasındaki korelasyonu araştırdı.

-Histogramlar ve çubuk grafikler kullanılarak görselleştirilmiş değişken dağılımları ve bunların hedef değişkenle ilişkileri.



STEP 2: ÖZELLİK MÜHENDİSLİĞİ

1.Değişken Eklemeler

-BMI (Vücut Kitle İndeksi) ve Su Tüketimi İndeksi (STI) hesaplandı ve veri setine eklendi.

2.⁠ ⁠Kodlama İşlemleri

-One-Hot Encoder kullanılarak kodlanmış nominal değişkenler.

-Label Encoder kullanılarak sıralı değişken kodlandı.

3.Standartlaştırma

-Standartlaştırılmış sayısal değişkenler.

4.Model Seçimi ve Eğitimi

-Lojistik Regresyon, Karar Ağaçları, Rastgele Orman, Destek Vektör Makineleri (SVM), Gradient Boosting, XGBoost ve LightGBM dahil olmak üzere çeşitli modeller seçildi ve eğitildi.

-Her model için hesaplanan doğruluk, F1 puanı ve ROC AUC puanı.

5.⁠ ⁠LightGBM Modelinin İncelenmesi

-LightGBM modelini daha detaylı inceledik.

-Modelin önemli özelliklerini ve katkılarını incelemek için Learning Curve ve feature importance kullanıldı.



STEP 3: NİHAİ MODELİN OLUŞTURULMASI VE TEST VERİLERİNİN HAZIRLANMASI

1.Nihai Modelin Oluşturulması

-En iyi performansa sahip LightGBM modeli seçildi.

-LightGBM modelini tüm veri kümesi üzerinde eğittik ve Stratified KFold çapraz doğrulamasını kullanarak performansını değerlendirdik.

2.Test Verilerinin Hazırlanması

-Eğitim veri setinde yapılan işlemlerin aynısı test veri setine de uygulandı.

-Test verilerindeki obezite seviyelerini tahmin etmek için modelimizi kullandık.



Bu adımlarla veri setini, tasarlanmış özellikleri, seçilen modelleri analiz ettik ve sonuçta obezite seviyelerini tahmin etmek için bir LightGBM modeli oluşturduk.



STEP :4 Kodumuzu Streamlit uygulamasına dönüştürdük ve kullanıcı arayüzü oluşturarak çalıştırdık



1.⁠ ⁠Adım: Gerekli Kitaplıkların İçe Aktarılması

Streamlit uygulamamız için gerekli kütüphaneleri import ederek işe başladık.



Adım 2: Modeli ve Veri Kümesini Yüklenmesi

Önceden eğitilmiş LightGBM modelini joblib kullanarak yükledik.



3.⁠ ⁠Adım: Farklı Sayfalar İçin İşlevlerin Tanımlanması

3.1 "Hakkımızda" Sayfası

Bu fonksiyon uygulama ve ekip üyeleri hakkında bilgiler içerir.

3.2 "Ana Sayfa" Sayfası

Bu fonksiyon uygulamanın ana sayfasıdır ve obezite hakkında bilgi verir.

3.3."Dinamik Grafikler" Sayfası

Bu işlev, yüklenen veri kümesine dayalı olarak Plotly Express'i kullanarak dinamik grafikleri görüntüler.

3.4."Obeziteyi Tahmin Etme" Sayfası

Bu işlev, kullanıcıların sağlık verilerini girmelerine ve önceden eğitilmiş modeli kullanarak obezite durumlarını tahmin etmelerine olanak tanır.



Adım 4: Ana İşlev ve Kolaylaştırılmış Yapılandırma

main() işlevi, kullanıcının kenar çubuğundan yaptığı seçime bağlı olarak farklı sayfaların oluşturulmasından sorumludur.


