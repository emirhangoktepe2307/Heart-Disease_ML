import streamlit as st
import pickle
import numpy as np
import os

# Sayfa yapılandırması
st.set_page_config(
    page_title="Kalp Hastalığı Tahmin Uygulaması",
    page_icon="❤️",
    layout="centered"
)

# Başlık
st.title("Kalp Hastalığı Tahmin Uygulaması")
st.write("Bu uygulama, verilen bilgilere göre kalp hastalığı riskini tahmin eder.")

# Model yükleme
@st.cache_resource
def load_model():
    try:
        model_path = 'heart_model.pkl'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None

# Model yükleme denemesi
model = load_model()
if model is None:
    st.error("Model yüklenemedi. Lütfen model dosyasının doğru konumda olduğundan emin olun.")
    st.stop()

# Kullanıcı girdileri
st.subheader("Lütfen aşağıdaki bilgileri giriniz:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Yaş", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
    cp = st.selectbox("Göğüs Ağrısı Tipi", ["Tipik Angina", "Atipik Angina", "Non-Anginal Ağrı", "Asemptomatik"])
    trestbps = st.number_input("Dinlenme Kan Basıncı (mm Hg)", min_value=90, max_value=200, value=120)
    chol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=200)

with col2:
    fbs = st.selectbox("Açlık Kan Şekeri > 120 mg/dl", ["Hayır", "Evet"])
    restecg = st.selectbox("EKG Sonuçları", ["Normal", "ST-T Dalga Anormalliği", "Sol Ventrikül Hipertrofisi"])
    thalach = st.number_input("Maksimum Kalp Atış Hızı", min_value=60, max_value=202, value=150)
    exang = st.selectbox("Egzersiz Kaynaklı Angina", ["Hayır", "Evet"])
    oldpeak = st.number_input("ST Depresyonu", min_value=0.0, max_value=6.2, value=0.0, step=0.1)

# Veri dönüşümleri
sex = 1 if sex == "Erkek" else 0
cp = ["Tipik Angina", "Atipik Angina", "Non-Anginal Ağrı", "Asemptomatik"].index(cp)
fbs = 1 if fbs == "Evet" else 0
restecg = ["Normal", "ST-T Dalga Anormalliği", "Sol Ventrikül Hipertrofisi"].index(restecg)
exang = 1 if exang == "Evet" else 0

# Tahmin butonu
if st.button("Tahmin Et"):
    try:
        # Girdileri diziye dönüştürme
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]])
        
        # Tahmin yapma
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Sonuçları gösterme
        st.subheader("Tahmin Sonucu")
        if prediction[0] == 1:
            st.error("Yüksek Kalp Hastalığı Riski")
        else:
            st.success("Düşük Kalp Hastalığı Riski")
        
        st.write(f"Risk Olasılığı: {probability[0][1]*100:.2f}%")
    except Exception as e:
        st.error(f"Tahmin yapılırken bir hata oluştu: {str(e)}")

# Bilgilendirme
st.markdown("---")
st.markdown("""
### Bilgilendirme
Bu uygulama sadece tahmin amaçlıdır ve tıbbi bir teşhis aracı değildir. 
Herhangi bir sağlık sorununuz için mutlaka bir sağlık uzmanına başvurunuz.
""") 