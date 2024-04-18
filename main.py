import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Modeli joblib ile yükle
model = joblib.load("BAU_Miuul_final_model.pkl")


def display_about():
    st.title('Hakkımızda')
    st.write('''
    Bu uygulama, kullanıcıların sağlık verileri üzerinden obezite durumunu tahmin etmek amacıyla geliştirilmiştir.
    Uygulama, çeşitli beslenme ve yaşam tarzı verilerini analiz ederek obezite riskini değerlendirmeye yardımcı olur.
    Projemiz, kullanıcıların sağlıklarını daha iyi anlamaları ve yönetmeleri için bilgilendirici içerikler sunmayı hedeflemektedir.
    ''')

    st.subheader('Takım Üyeleri')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('Asli.jpeg', width=150)
        st.markdown('**Aslı Öztürk**')
        st.markdown('[LinkedIn Profili](https://www.linkedin.com/in/ozturk-asli/)')
        st.write("Aslı son 11 yılda 20'den fazla yazılım geliştirme projesinde , farklı kurumsal firmalarda Business Analyst olarak çalıştı.Veri bilimi ve makine öğrenmesinde kendini geliştirerek kariyerine Data Analyst/Data Scientist olarak devam etmek istiyor.")

    with col2:
        st.image('Begum.jpeg', width=150)
        st.markdown('**Begüm Baybora**')
        st.markdown('[LinkedIn Profili](https://www.linkedin.com/in/begumbaybora/)')
        st.write("Begüm, supply chain alanında tecrübeli ve şimdi veri analizi ile yapay zeka konularına ilgi duyuyor. Bu alanlarda kendini geliştirerek, supply chain yönetiminde veri odaklı ve yapay zeka destekli çözümler geliştirmeye hevesli.")

    with col3:
        st.image('Ugur.jpeg', width=150)
        st.markdown('**Uğur Can Odabaşı**')
        st.markdown('[LinkedIn Profili](https://www.linkedin.com/in/ugurcanodabasi)')
        st.write("Uğur, çeşitli endüstrilerde teknoloji ve finans çözümleri sunarak teknik liderlik yapmaktadır. Şimdi ise veri bilimi ve makine öğrenmesi alanında kendini geliştirerek bu başarılarına yeni boyutlar eklemeyi hedefliyor.")

# Anasayfa
def home_page():
    st.title('Anasayfa')

    # Obezite oranlarını gösteren dünya haritası (yer tutucu olarak statik bir görsel)
    st.image('Obesity_rate_(WHO,_2022).png', caption='Dünya Obezite Haritası')

    # Obezitenin çağın hastalığı olduğunu anlatan metin
    st.write("""
    ## Obezite: Çağımızın Hastalığı

    Obezite, dünya genelinde milyonlarca insanı etkileyen ve hızla yayılan ciddi bir sağlık sorunudur. 
    Vücut kitle indeksi (VKİ) 30'un üzerinde olan bireyler obez olarak sınıflandırılır ve bu durum, 
    çeşitli kronik hastalıkların riskini önemli ölçüde artırır. Obezite; kalp hastalıkları, tip 2 diyabet, 
    bazı kanser türleri ve kas-iskelet sistemi bozuklukları gibi sağlık sorunlarına yol açabilir.

    Obeziteyle mücadele, sadece bireysel değil, aynı zamanda küresel bir çaba gerektirir. 
    Sağlıklı beslenme alışkanlıkları edinmek, düzenli fiziksel aktivite yapmak ve sağlıklı yaşam tarzı seçimleri,
    obeziteyle mücadelede kilit rol oynar. Toplumlar ve hükümetler, sağlıklı gıdalara erişimi kolaylaştırmak, 
    fiziksel aktiviteyi teşvik etmek ve sağlık eğitimini artırmak için politikalar geliştirmeli ve uygulamalıdır.
    """)
# Dinamik Grafikler
def dynamic_graphs():
    st.title('Veri ve Modelimiz Hakkında')

    st.markdown("""
    ## Değişkenler

    | Numerik Değişkenler | Kategorik Değişkenler |
    | ------------------- | --------------------- |
    | **id**              | **Gender (Cinsiyet)** |
    | **Age (Yaş)**       | **family_history_with_overweight (Ailede fazla kilolu birey olup olmaması)** |
    | **Height (Boy)**    | **FAVC (Sıklıkla yüksek kalorili yiyecek tüketimi)** |
    | **Weight (Kilo)**   | **CAEC (Yemek arası atıştırmalıklar)** |
    | **FCVC (Sıklıkla taze sebze ve meyve tüketimi)** | **SMOKE (Sigara kullanımı)** |
    | **NCP (Ana öğün sayısı)** | **SCC (Kalori sayımı)** |
    | **CH2O (Günlük su tüketimi)** | **CALC (Alkol tüketimi)** |
    | **FAF (Fiziksel aktivite sıklığı)** | **MTRANS (Taşıt kullanım türü)** |
    | **TUE (Teknoloji kullanım süresi)** | **NObeyesdad (Obezite seviyesi)** |
    """)

# Fotoğrafı yüklemek için
    image_path = "LearningCurve.png"  # 'image.jpg' yerine yüklemek istediğiniz dosyanın yolunu yazın.
    image = st.image(image_path, caption='Learning Curve', use_column_width=True)

# Fotoğraf altına bilgi eklemek için
    st.write("Bu eğitim eğrisi, modelinizin hem eğitim setindeki performansını hem de çapraz doğrulama setindeki performansını eğitim setinin büyüklüğüne göre gösteriyor.")

# Fotoğrafı yüklemek için
    image_path = "features.png"  # 'image.jpg' yerine yüklemek istediğiniz dosyanın yolunu yazın.
    image = st.image(image_path, caption='Değişkenlerin Modele Etkisi', use_column_width=True)

# Fotoğraf altına bilgi eklemek için
    st.write("""
    Girdilerin modelimizin tahminlemeye etkisini gösteriyor.

    - Su Tüketimi İndeksi (STI): `STI = CH2O * FAF`
    - Beden Kitle İndeksi (BMI): `BMI = Weight / (Height**2)`
    """)



import numpy as np
import pandas as pd

# Obezite Tahmini
def predict_obesity():
    st.title('Obezite Tahmini Yap')

    # Kullanıcıdan alınacak girdiler
    gender = st.selectbox('Cinsiyetiniz', ['Erkek', 'Kadın'])
    age = st.number_input('Yaşınız', min_value=1, max_value=100, value=25)
    height = st.number_input('Boyunuz (cm)', min_value=50, max_value=250, value=170)
    weight = st.number_input('Kilonuz (kg)', min_value=20, max_value=300, value=70)
    family_history = st.selectbox('Ailenizde obezite var mı?', ['Evet', 'Hayır'])
    favc = st.selectbox('Yüksek kalorili yiyecek sık tüketir misiniz?', ['Evet', "Hayır"])
    fcvc = st.selectbox('Sebze tüketim sıklığınız?', ['Hiç', 'Ara sıra', 'Sık'])
    ncp = st.selectbox('Günlük ana öğün sayınız?', ['1', '2', '3', '4 veya daha fazla'])
    caec = st.selectbox('Günde ara öğün yapar mısın?', ['Asla', 'Bazen', 'Sıklıkla', 'Her zaman'])
    smoke = st.selectbox('Sigara kullanımınız?', ['Evet', 'Hayır'])
    ch2o = st.slider('Günlük su tüketiminiz (litre)', 1.0, 3.0, 1.0)
    scc = st.selectbox('Kalori tüketimini takip ediyor musunuz?', ['Evet', 'Hayır'])
    faf = st.selectbox("Haftalık fiziksel aktivite sıklığınız?", ['Asla', 'Bazen', 'Sıklıkla', 'Her zaman'])
    tue = st.selectbox('Günlük teknoloji kullanım sıklığınız?', ['Çok az', 'Az', 'Orta', 'Sık', "Çok sık"])
    calc = st.selectbox('Alkol tüketim sıklığınız?', ['Asla', 'Bazen', 'Sıklıkla'])
    mtrans = st.selectbox('Genel ulaşım şekliniz?', ['Yürüyerek', 'Bisiklet', 'Toplu taşıma', 'Araba', 'Motosiklet'])

    # Kategorik değişkenleri sayısal değerlere dönüştür
    gender = 1 if gender == 'Erkek' else 0
    family_history = 1 if family_history == 'Evet' else 0
    favc = 1 if favc == 'Evet' else 0
    fcvc_mapping = {'Hiç': 1, 'Ara sıra': 2, 'Sık': 3}
    fcvc = fcvc_mapping[fcvc]
    ncp_mapping = {'1': 1, '2': 2, '3': 3, '4 veya daha fazla': 4}
    ncp = ncp_mapping[ncp]
    caec_mapping = {'Asla': 0, 'Bazen': 1, 'Sıklıkla': 2, 'Her zaman': 3}
    caec = caec_mapping[caec]
    smoke = 1 if smoke == 'Evet' else 0
    scc = 1 if scc == 'Evet' else 0
    faf_mapping = {'Asla': 0, 'Bazen': 1, 'Sıklıkla': 2, 'Her zaman': 3}
    faf = faf_mapping[faf]
    tue_mapping = {'Çok az': 0, 'Az': 0.5, 'Orta': 1, 'Sık': 1.5, 'Çok sık': 2}
    tue = tue_mapping[tue]
    calc_mapping = {'Asla': 0, 'Bazen': 1, 'Sıklıkla': 2}
    calc = calc_mapping[calc]
    mtrans_mapping = {'Yürüyerek': 0, 'Bisiklet': 1, 'Toplu taşıma': 2, 'Araba': 3, 'Motosiklet': 4}
    mtrans = mtrans_mapping[mtrans]

    # Toplam girdi vektörünü oluştur
    input_data = np.array(
        [gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans])


    # Tüm değişkenleri içeren girdi matrisini oluşturun ve eksik değerleri ekleyin (Eğer varsa)
    if input_data.shape[0] < 24:
        additional_data = np.full((24 - input_data.shape[0],), np.nan)  # Eksik özellikler için NaN doldur
        input_data = np.concatenate((input_data, additional_data))

    if st.button('Tahmin Et'):
        prediction = model.predict(input_data.reshape(1, -1))
        st.subheader(f'Tahmin edilen obezite durumu: {prediction[0]}')



# Diğer fonksiyonlar ve Streamlit yapılandırması

# Streamlit uygulamasını yapılandır
def main():
    st.sidebar.title('Navigasyon')
    page = st.sidebar.radio('Sayfayı Seçin:', ['Hakkımızda', 'Anasayfa', 'Dinamik Grafikler', 'Obezite Tahmini'])

    if page == 'Hakkımızda':
        display_about()
    elif page == 'Anasayfa':
        home_page()
    elif page == 'Dinamik Grafikler':
        dynamic_graphs()
    elif page == 'Obezite Tahmini':
        predict_obesity()


if __name__ == "__main__":
    main()
