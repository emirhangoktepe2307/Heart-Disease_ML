import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

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
            'age', 'sex', 'trestbps', 'chol', 'exercise_enc', 
            'smoking_enc', 'fhd_enc', 'diabetes_enc', 'bmi', 
            'high_blo_pre_enc', 'hdl_enc', 'ldl_enc', 
            'alcohol_enc', 'stress_enc', 'sleep_hours', 
            'sugar_cons_enc', 'trglycrde_lvl', 'fbs', 
            'crp_lvl', 'hmocystesine_lvl'
        ])
    
    # Triglyceride seviyesini kategorize et
    X['Ves_Hardness'] = X['trglycrde_lvl'].apply(categorize_triglyceride)
    
    # Kan Basıncı Ve Enfeksiyon Oranı
    X['bp_crp_ratio'] = X['crp_lvl'].astype(float) / X['trestbps'].astype(float)
    
    # Kolesterol ve Kan Basıncı Oranı
    X['ves_dia_est'] = X['trestbps'].astype(float) / X['chol'].astype(float)
    
    # Yemek Skoru (Skor Ne Kadar Yüksekse Beslenme Düzeni O Kadar İyi)
    X['meal_order_record'] = X['chol'].astype(float) / X['bmi'].astype(float)
    
    # Egzersiz Durumuna Bağlı Kolesterol Oranı
    X['chol_exe_ratio'] = X['chol'].astype(float) / X['exercise_enc'].astype(float)
    
    # Sütun sırasını modelin beklediği sıraya göre düzenle
    X = X[[
        'age', 'sex', 'trestbps', 'chol', 'exercise_enc', 
        'smoking_enc', 'fhd_enc', 'diabetes_enc', 'bmi', 
        'high_blo_pre_enc', 'hdl_enc', 'ldl_enc', 
        'alcohol_enc', 'stress_enc', 'sleep_hours', 
        'sugar_cons_enc', 'trglycrde_lvl', 'fbs', 
        'crp_lvl', 'hmocystesine_lvl', 'Ves_Hardness',
        'bp_crp_ratio', 'ves_dia_est', 'meal_order_record',
        'chol_exe_ratio'
    ]]
    
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
        # Tam yolu kullan
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'heart_pipeline.pkl')
        
        st.write(f"Model dosyası aranıyor: {model_path}")
        
        if not os.path.exists(model_path):
            st.error(f"Model dosyası bulunamadı: {model_path}")
            # Mevcut dizindeki dosyaları listele
            st.write("Mevcut dizindeki dosyalar:")
            for file in os.listdir(current_dir):
                st.write(f"- {file}")
            return None
            
        with open(model_path, 'rb') as file:
            try:
                model = pickle.load(file)
                st.success("Model başarıyla yüklendi!")
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

# Model tipini ve özelliklerini kontrol et
st.write("Model tipi:", type(model))
if hasattr(model, 'feature_names_in_'):
    st.write("Model özellikleri:", model.feature_names_in_)
else:
    st.warning("Model özellik isimleri bulunamadı!")

# Kullanıcı girdileri
st.subheader("Lütfen aşağıdaki bilgileri giriniz:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Yaş", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
    trestbps = st.number_input("Dinlenme Kan Basıncı (mm Hg)", min_value=90, max_value=200, value=110)
    chol = st.number_input("Kolesterol (mg/dl) Seviyesini Giriniz:", min_value=100, max_value=600, value=200)
    bmi = st.number_input("Vücut Kitle İndeksinizi Giriniz:", min_value=10, max_value=50, value=20)
    fbs = st.number_input("Açlık Kan Şekeri Değerinizi Giriniz:", min_value=50, max_value=100, value=20)
    sleep_hours=st.number_input("Rutin Uyku Saatinizi (Ortalama) Giriniz:", min_value=2, max_value=14, value=7)
    trglycrde_lvl=st.number_input("Kan Tahlilinizde Saptanan Trigliserit Değerini Giriniz",min_value=100,max_level=400,value=250)
    crp_lvl=st.number_input("Kan Tahlilinizde Saptanan Enfeksiyon (CRP) Değerinizi Giriniz",min_value=1,max_level=14.99,value=5.1)
    hmocystesine_lvl=st.number_input("Kan Tahlilinizde Ölçülen Homosistein Seviyesi (Hcy) Değerini Giriniz",min_value=5,max_value=19.99,value=6.5)

with col2:
    
    thalach = st.number_input("Maksimum Kalp Atış Hızı", min_value=60, max_value=202, value=150)
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
sugar_cons_enc={"Az":0,"Orta":1,"Çok":2}

# Tahmin butonu
if st.button("Tahmin Et"):
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
        
        # Model özelliklerini kontrol et
        st.write("Model özellikleri:", model.feature_names_in_)
        st.write("Girdi özellikleri:", input_df.columns.tolist())
        
        # Tahminleme
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        # Sonuçları gösterme
        st.subheader("Tahmin Sonucu")
        if prediction[0] == 1:
            st.error("Yüksek Kalp Hastalığı Riski")
        else:
            st.success("Düşük Kalp Hastalığı Riski")
        
        st.write(f"Risk Olasılığı: {probability[0][1]*100:.2f}%")
    except Exception as e:
        st.error(f"Tahmin yapılırken bir hata oluştu: {str(e)}")
        st.write("Hata detayı:", str(e))
        st.write("Model tipi:", type(model))
        if hasattr(model, 'feature_names_in_'):
            st.write("Model özellikleri:", model.feature_names_in_)

# Bilgilendirme
st.markdown("---")
st.markdown("""
### Bilgilendirme
Bu uygulama sadece tahmin amaçlıdır ve tıbbi bir teşhis aracı değildir. 
Herhangi bir sağlık sorununuz için mutlaka bir sağlık uzmanına başvurunuz.
""") 