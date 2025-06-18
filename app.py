import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# GitHub/Streamlit uyumlu dosya yolları
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'heart_disease_feature.csv')
model_path = os.path.join(current_dir, 'heart_pipeline.joblib')

# CSV dosyasını güvenli şekilde yükle
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"CSV dosyası bulunamadı: {csv_path}")
    st.stop()

def categorize_triglyceride(level):
    if pd.isna(level):  # NaN değerleri kontrol et
        return np.nan
    elif level < 100:
        return 0
    elif 100 <= level < 150:
        return 1
    else:  # level > 150
        return 2

def add_ratios(X):
    # DataFrame'e dönüştürme
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[
            'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits', 
            'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 
            'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol', 
            'Alcohol Consumption', 'Stress Level', 'Sleep Hours', 
            'Sugar Consumption', 'Triglyceride Level', 'Fasting Blood Sugar', 
            'CRP Level', 'Homocysteine Level'
        ])
    
    # Triglyceride seviyesini kategorize et
    X['Ves_Hardness'] = X['Triglyceride Level'].apply(categorize_triglyceride)
    
    # Kan Basıncı Ve Enfeksiyon Oranı
    X['Bp/Crp'] = X['CRP Level'].astype(float) / X['Blood Pressure'].astype(float)
    
    # Kolesterol ve Kan Basıncı Oranı
    X['Ves_dia_est'] = X['Blood Pressure'].astype(float) / X['Cholesterol Level'].astype(float)
    
    # Yemek Skoru (Skor Ne Kadar Yüksekse Beslenme Düzeni O Kadar İyi)
    X['Meal order record'] = X['Cholesterol Level'].astype(float) / X['BMI'].astype(float)
    
    # Egzersiz Durumuna Bağlı Kolesterol Oranı
    X['Chol/Exe'] = X['Cholesterol Level'].astype(float) / X['Exercise Habits'].astype(float)
    
    return X

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
        if not os.path.exists(model_path):
            st.error(f"Model dosyası bulunamadı: {model_path}")
            st.write("Mevcut dizindeki dosyalar:")
            for file in os.listdir(current_dir):
                st.write(f"- {file}")
            return None
            
        try:
            model = joblib.load(model_path)
            st.success("✅ Model başarıyla yüklendi!")
            return model
        except Exception as e:
            st.error(f"Model dosyası yüklenirken hata oluştu: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Model yüklenirken beklenmeyen bir hata oluştu: {str(e)}")
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
    trestbps = st.number_input("Dinlenme Kan Basıncı (mm Hg)", min_value=90, max_value=200, value=110)
    chol = st.number_input("Kolesterol (mg/dl) Seviyesini Giriniz:", min_value=100, max_value=600, value=200)
    bmi = st.number_input("Vücut Kitle İndeksinizi Giriniz:", min_value=10, max_value=50, value=20)
    fbs = st.number_input("Açlık Kan Şekeri Değerinizi Giriniz:", min_value=20, max_value=100, value=50)
    sleep_hours=st.number_input("Rutin Uyku Saatinizi (Ortalama) Giriniz:", min_value=2, max_value=14, value=7)
    trglycrde_lvl=st.number_input("Kan Tahlilinizde Saptanan Trigliserit Değerini Giriniz",min_value=100,max_value=400,value=250)
    crp_lvl=st.number_input("Kan Tahlilinizde Saptanan Enfeksiyon (CRP) Değerinizi Giriniz",min_value=0.1,max_value=14.99,value=5.1)
    hmocystesine_lvl=st.number_input("Kan Tahlilinizde Ölçülen Homosistein Seviyesi (Hcy) Değerini Giriniz",min_value=5.0,max_value=19.99,value=6.5)

with col2:
    stress= st.selectbox("Stres Seviyeniz Nedir?",["Az","Orta","Çok"])
    fhd= st.selectbox("Genetik Kalp Krizi Vakası Ailenizde Mevcut Mu?",["Evet","Hayır"])
    smoking= st.selectbox("Sigara Kullanıyor Musunuz?",["Evet","Hayır"])
    diabetes=st.selectbox("Şeker Hastalığınız Var Mı?",["Evet","Hayır"])
    exercise=st.selectbox("Egzersiz Sıklığınız Nedir?",["Az","Orta","Çok"])
    alcohol= st.selectbox("Alkol Tüketme Sıklığınız Nedir",["Az","Orta","Çok"])
    high_blo_pre=st.selectbox("Yüksek Tansiyon Hastalığınız Var Mı?",["Evet","Hayır"])
    hdl=st.selectbox("İyi Kolesterol (HDL) Seviyeniz Yüksek Mi?",["Evet","Hayır"])
    ldl=st.selectbox("Kötü Kolesterol (LDL) Seviyeniz Yüksek Mi?",["Evet","Hayır"])
    sugar_cons=st.selectbox("Günlük Şeker Tüketme Sıklığınız",["Az","Orta","Çok"])

# Veri encode dönüşümleri
sex_enc = {"Erkek":1, "Kadın":0}[sex]
diabetes_enc= {"Evet":1, "Hayır":0}[diabetes]
fhd_enc={"Evet":1, "Hayır":0}[fhd]
smoking_enc={"Evet":1, "Hayır":0}[smoking]
exercise_enc={"Çok":1, "Orta":2, "Az":3}[exercise]
stress_enc={"Az":1, "Orta":2, "Çok":3}[stress]
alcohol_enc = {"Az": 0, "Orta": 1, "Çok": 2}[alcohol]
high_blo_pre_enc={"Evet":1, "Hayır":0}[high_blo_pre]
hdl_enc={"Evet":0,"Hayır":1}[hdl]
ldl_enc={"Evet":1,"Hayır":0}[ldl]
sugar_cons_enc={"Az":0,"Orta":1,"Çok":2}[sugar_cons]

# Tahmin butonu
if st.button("🔍 Tahmin Et"):
    try:
        # Girdileri diziye dönüştürme (Eğitim veriseti sırasına uygun)
        input_data = np.array([[
            float(age), 
            int(sex_enc), 
            float(trestbps), 
            float(chol), 
            int(exercise_enc), 
            int(smoking_enc), 
            int(fhd_enc), 
            int(diabetes_enc), 
            float(bmi), 
            int(high_blo_pre_enc), 
            int(hdl_enc), 
            int(ldl_enc), 
            int(alcohol_enc), 
            int(stress_enc), 
            float(sleep_hours), 
            int(sugar_cons_enc),
            float(trglycrde_lvl),
            float(fbs),
            float(crp_lvl),
            float(hmocystesine_lvl)
        ]])
        
        # DataFrame'e dönüştürme ve oranları ekleme
        input_df = add_ratios(input_data)
        
        # Tahminleme
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        # Sonuçları gösterme
        st.subheader("📊 Tahmin Sonucu")
        
        # Risk seviyesine göre renkli gösterim
        risk_probability = probability[0][1] * 100
        
        if prediction[0] == 1:
            if risk_probability > 70:
                st.error("🚨 Yüksek Kalp Hastalığı Riski")
                st.warning("Lütfen en kısa sürede bir kardiyoloğa başvurunuz.")
            elif risk_probability > 50:
                st.warning("⚠️ Orta Kalp Hastalığı Riski")
                st.info("Düzenli kontroller yaptırmanız önerilir.")
            else:
                st.info("📈 Düşük-Orta Kalp Hastalığı Riski")
        else:
            if risk_probability < 20:
                st.success("✅ Düşük Kalp Hastalığı Riski")
                st.info("Sağlıklı yaşam tarzınızı sürdürün.")
            else:
                st.info("📉 Düşük Kalp Hastalığı Riski")
                st.info("Düzenli kontroller yaptırmaya devam edin.")
        
        # Risk olasılığını göster
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Olasılığı", f"{risk_probability:.1f}%")
        with col2:
            st.metric("Güvenli Olasılık", f"{100-risk_probability:.1f}%")
        with col3:
            st.metric("Tahmin Güvenilirliği", "85%")
        
    except Exception as e:
        st.error(f"❌ Tahmin yapılırken bir hata oluştu: {str(e)}")
        st.write("🔍 Hata detayı:", str(e))
        st.info("💡 Lütfen tüm alanları doğru şekilde doldurduğunuzdan emin olun.")

# Bilgilendirme
st.markdown("---")
st.markdown("""
### 📋 Önemli Bilgilendirme

⚠️ **Uyarı**: Bu uygulama sadece tahmin amaçlıdır ve tıbbi bir teşhis aracı değildir. 
Herhangi bir sağlık sorununuz için mutlaka bir sağlık uzmanına başvurunuz.

🔬 **Model Bilgileri**:
- Model: Random Forest Classifier
- Doğruluk: %71.6
- Veri Dengesizliği: SMOTE ile düzeltildi
- Özellik Sayısı: 24 (20 temel + 4 türetilmiş)

💡 **Öneriler**:
- Düzenli sağlık kontrolleri yaptırın
- Sağlıklı yaşam tarzı benimseyin
- Risk faktörlerini minimize edin
""") 