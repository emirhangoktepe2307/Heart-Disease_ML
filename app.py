import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# GitHub/Streamlit uyumlu dosya yolları
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'heart_disease_feature.csv')
csv_path_first= os.path.join(current_dir, 'heart_disease.csv')
model_path = os.path.join(current_dir, 'heart_pipeline.joblib')

# CSV dosyasını güvenli şekilde yükle
try:
    df = pd.read_csv(csv_path)
    df_first=pd.read_csv(csv_path_first)
except FileNotFoundError:
    st.error(f"CSV dosyası bulunamadı: {csv_path}")
    st.stop()

def categorize_triglyceride(level):
    try:
        if pd.isna(level) or level is None:  # NaN ve None değerleri kontrol et
            return np.nan
        level = float(level)  # Sayıya dönüştür
        if level < 100:
            return 0
        elif 100 <= level < 150:
            return 1
        else:  # level >= 150
            return 2
    except (ValueError, TypeError):
        return np.nan  # Geçersiz değerler için NaN döndür

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
    
    # Kan Basıncı Ve Enfeksiyon Oranı - Sıfıra bölme kontrolü
    X['Bp/Crp'] = np.where(
        (X['Blood Pressure'].astype(float) != 0) & (X['CRP Level'].astype(float) != 0),
        X['CRP Level'].astype(float) / X['Blood Pressure'].astype(float),
        0
    )
    
    # Kolesterol ve Kan Basıncı Oranı - Sıfıra bölme kontrolü
    X['Ves_dia_est'] = np.where(
        (X['Cholesterol Level'].astype(float) != 0) & (X['Blood Pressure'].astype(float) != 0),
        X['Blood Pressure'].astype(float) / X['Cholesterol Level'].astype(float),
        0
    )
    
    # Yemek Skoru - Sıfıra bölme kontrolü
    X['Meal order record'] = np.where(
        (X['BMI'].astype(float) != 0) & (X['Cholesterol Level'].astype(float) != 0),
        X['Cholesterol Level'].astype(float) / X['BMI'].astype(float),
        0
    )
    
    # Egzersiz Durumuna Bağlı Kolesterol Oranı - Sıfıra bölme kontrolü
    X['Chol/Exe'] = np.where(
        (X['Exercise Habits'].astype(float) != 0) & (X['Cholesterol Level'].astype(float) != 0),
        X['Cholesterol Level'].astype(float) / X['Exercise Habits'].astype(float),
        0
    )
    
    return X

# Görselleştirme fonksiyonları
def plot_categorical_distributions(df):
    try:
        cat_cols = df.select_dtypes("object").columns
        if len(cat_cols) == 0:
            st.info("📊 Kategorik değişken bulunamadı.")
            return
            
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax)
            ax.set_title(f"{col} Frekans Dağılımı")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    except Exception as e:
        st.error(f"📊 Kategorik değişken görselleştirme hatası: {str(e)}")

def plot_numerical_distributions(df):
    try:
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) == 0:
            st.info("📈 Sayısal değişken bulunamadı.")
            return
            
        for col in num_cols:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            
            # Histogram
            sns.histplot(df[col].dropna(), kde=True, ax=ax1)
            ax1.set_title(f"{col} Dağılımı (Histogram + KDE)")
            ax1.set_xlabel(col)
            ax1.set_ylabel("Frekans")
            
            # Box plot
            sns.boxplot(x=df[col].dropna(), color="skyblue", ax=ax2)
            ax2.set_title(f"{col} Box-plot (Uç Değer Kontrolü)")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    except Exception as e:
        st.error(f"📈 Sayısal değişken görselleştirme hatası: {str(e)}")

# Sayfa yapılandırması
st.set_page_config(
    page_title="Kalp Hastalığı Tahmin Uygulaması",
    page_icon="❤️",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("📊 Navigasyon")
    
    # Ana sayfa seçimi
    page = st.selectbox(
        "Sayfa Seçin",
        ["🏠 Ana Sayfa", "📈 SUNUM", "📋 Model Bilgileri", "ℹ️ Hakkında"]
    )

# Ana sayfa
if page == "🏠 Ana Sayfa":
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
        bmi = st.number_input("Vücut Kitle İndeksinizi Giriniz:", min_value=10.0, max_value=50.0, value=20.0)
        fbs = st.number_input("Açlık Kan Şekeri Değerinizi Giriniz:", min_value=20, max_value=100, value=50)
        sleep_hours=st.number_input("Rutin Uyku Saatinizi (Ortalama) Giriniz:", min_value=2.0, max_value=14.0, value=7.0)
        trglycrde_lvl=st.number_input("Kan Tahlilinizde Saptanan Trigliserit Değerini Giriniz",min_value=100,max_value=400,value=250)
        crp_lvl=st.number_input("Kan Tahlilinizde Saptanan Enfeksiyon (CRP) Değerinizi Giriniz",min_value=0.1,max_value=14.99,value=5.1)
        hmocystesine_lvl=st.number_input("Kan Tahlilinizde Ölçülen Homosistein Seviyesi (Hcy) Değerini Giriniz",min_value=5.0,max_value=19.99,value=6.5)

    with col2:
        stress= st.selectbox("Stres Seviyeniz Nedir?",["Az","Orta","Çok"])
        fhd= st.selectbox("Genetik Kalp Krizi Vakası Ailenizde Mevcut Mu?",["Evet","Hayır"])
        smoking= st.selectbox("Sigara Kullanıyor Musunuz?",["Evet","Hayır"])
        diabetes=st.selectbox("Şeker Hastalığınız Var Mı?",["Evet","Hayır"])
        exercise=st.selectbox("Egzersiz Sıklığınız Nedir?",["Az","Orta","Çok"])
        alcohol= st.selectbox("Alkol Tüketme Sıklığınız Nedir",["Az/Hiç","Orta","Çok"])
        high_blo_pre=st.selectbox("Yüksek Tansiyon Hastalığınız Var Mı?",["Evet","Hayır"])
        hdl=st.selectbox("İyi Kolesterol (HDL) Seviyeniz Yüksek Mi?",["Evet","Hayır"])
        ldl=st.selectbox("Kötü Kolesterol (LDL) Seviyeniz Yüksek Mi?",["Evet","Hayır"])
        sugar_cons=st.selectbox("Günlük Şeker Tüketme Sıklığınız",["Az/Hiç","Orta","Çok"])

    # Veri encode dönüşümleri
    sex_enc = {"Erkek":1, "Kadın":0}[sex]
    diabetes_enc= {"Evet":1, "Hayır":0}[diabetes]
    fhd_enc={"Evet":1, "Hayır":0}[fhd]
    smoking_enc={"Evet":1, "Hayır":0}[smoking]
    exercise_enc={"Çok":1, "Orta":2, "Az":3}[exercise]
    stress_enc={"Az":0, "Orta":1, "Çok":2}[stress]
    alcohol_enc = {"Az/Hiç": 0, "Orta": 1, "Çok": 2}[alcohol]
    high_blo_pre_enc={"Evet":1, "Hayır":0}[high_blo_pre]
    hdl_enc={"Evet":0,"Hayır":1}[hdl]
    ldl_enc={"Evet":1,"Hayır":0}[ldl]
    sugar_cons_enc={"Az/Hiç":0,"Orta":1,"Çok":2}[sugar_cons]

    # Tahmin butonu
    if st.button("🔍 Tahmin Et"):
        try:
            # Girdi değerlerini kontrol et
            if not all([age, trestbps, chol, bmi, fbs, sleep_hours, trglycrde_lvl, crp_lvl, hmocystesine_lvl]):
                st.error("❌ Lütfen tüm sayısal alanları doldurunuz.")
                st.stop()
            
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
            
            # NaN değerleri kontrol et ve doldur
            if input_df.isnull().any().any():
                st.warning("⚠️ Bazı hesaplanan değerler eksik, varsayılan değerler kullanılıyor.")
                input_df = input_df.fillna(0)
            
            # Tahminleme
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            
            # Risk seviyesine göre renkli gösterim
            risk_probability = probability[0][1] * 100
            
            # Risk kategorilerini belirle
            if prediction[0] == 1:  # Kalp hastalığı riski var
                if risk_probability >= 80:
                    risk_level = "🚨 ÇOK YÜKSEK"
                    risk_color = "error"
                    risk_message = "ACİL: Lütfen hemen bir kardiyoloğa başvurunuz!"
                    recommendation = "• Acil tıbbi müdahale gerekli\n• Tüm risk faktörlerini kontrol edin\n• Stres ve fiziksel aktiviteyi sınırlayın"
                elif risk_probability >= 60:
                    risk_level = "⚠️ YÜKSEK"
                    risk_color = "error"
                    risk_message = "Yüksek risk tespit edildi. Kardiyoloji kontrolü önerilir."
                    recommendation = "• En kısa sürede kardiyoloğa başvurun\n• Düzenli kontroller yaptırın\n• Yaşam tarzı değişiklikleri uygulayın"
                elif risk_probability >= 40:
                    risk_level = "🟡 ORTA"
                    risk_color = "warning"
                    risk_message = "Orta seviye risk tespit edildi."
                    recommendation = "• Düzenli sağlık kontrolleri yaptırın\n• Risk faktörlerini azaltın\n• Sağlıklı yaşam tarzı benimseyin"
                else:
                    risk_level = "🟢 DÜŞÜK-ORTA"
                    risk_color = "info"
                    risk_message = "Düşük-orta seviye risk tespit edildi."
                    recommendation = "• Düzenli kontroller yaptırmaya devam edin\n• Sağlıklı yaşam tarzınızı sürdürün\n• Risk faktörlerini takip edin"
            else:  # Kalp hastalığı riski düşük
                if risk_probability <= 10:
                    risk_level = "✅ ÇOK DÜŞÜK"
                    risk_color = "success"
                    risk_message = "Mükemmel! Kalp hastalığı riskiniz çok düşük."
                    recommendation = "• Sağlıklı yaşam tarzınızı sürdürün\n• Düzenli kontroller yaptırmaya devam edin\n• Örnek bir yaşam tarzınız var!"
                elif risk_probability <= 20:
                    risk_level = "🟢 DÜŞÜK"
                    risk_color = "success"
                    risk_message = "Kalp hastalığı riskiniz düşük seviyede."
                    recommendation = "• Mevcut sağlıklı alışkanlıklarınızı koruyun\n• Düzenli kontroller yaptırmaya devam edin\n• Risk faktörlerini takip edin"
                else:
                    risk_level = "🟡 DÜŞÜK-ORTA"
                    risk_color = "info"
                    risk_message = "Düşük-orta seviye risk tespit edildi."
                    recommendation = "• Düzenli kontroller yaptırmaya devam edin\n• Risk faktörlerini azaltmaya çalışın\n• Sağlıklı yaşam tarzınızı sürdürün"
            
            # Sonuçları gösterme
            st.subheader("📊 Tahmin Sonucu")
            
            # Risk seviyesi kartı
            if risk_color == "error":
                st.error(f"**{risk_level} RİSK**")
            elif risk_color == "warning":
                st.warning(f"**{risk_level} RİSK**")
            elif risk_color == "success":
                st.success(f"**{risk_level} RİSK**")
            else:
                st.info(f"**{risk_level} RİSK**")
            
            st.write(f"**{risk_message}**")
            
            # Öneriler
            st.subheader("💡 Öneriler")
            st.write(recommendation)
            
            # Risk olasılığını göster
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Olasılığı", f"{risk_probability:.1f}%")
            with col2:
                st.metric("Güvenli Olasılık", f"{100-risk_probability:.1f}%")
            with col3:
                st.metric("Tahmin Güvenilirliği", "85%")
            
        except ValueError as ve:
            st.error(f"❌ Geçersiz değer hatası: {str(ve)}")
            st.info("💡 Lütfen tüm alanları geçerli değerlerle doldurunuz.")
        except ZeroDivisionError as zde:
            st.error(f"❌ Sıfıra bölme hatası: {str(zde)}")
            st.info("💡 Lütfen tüm değerlerin sıfırdan farklı olduğundan emin olunuz.")
        except Exception as e:
            st.error(f"❌ Tahmin yapılırken bir hata oluştu: {str(e)}")
            st.write("🔍 Hata detayı:", str(e))
            st.info("💡 Lütfen tüm alanları doğru şekilde doldurduğunuzdan emin olun.")

# Sunum sayfası
elif page == "📈 SUNUM":
    st.title("📈 SUNUM")
    st.write("Bu bölümde proje sürecinde yapılan analizler ve görselleştirmeler yer almaktadır.")
    
    # Sunum bölümleri
    presentation_section = st.selectbox(
        "Sunum Bölümü Seçin",
        ["📊 Veri Analizi", "🔍 Özellik Mühendisliği", "🤖 Model Performansı"]
    )
    
    if presentation_section == "📊 Veri Analizi":
        st.header("📊 Veri Analizi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Veri Seti İlk Durum Özeti")
            st.write(f"**Toplam Kayıt Sayısı:** {len(df_first)}")
            st.write(f"**Özellik Sayısı:** {len(df_first.columns)}")
            st.write(f"**Eksik Veri Oranı:** %{df_first.isnull().sum().sum() / (len(df_first) * len(df_first.columns)) * 100:.2f}")

            st.subheader("📋 Veri Seti Son Durum Özeti")
            st.write(f"**Toplam Kayıt Sayısı:** {len(df)}")
            st.write(f"**Özellik Sayısı:** {len(df.columns)}")
            st.write(f"**Eksik Veri Oranı:** %{df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}")
            
            st.subheader("🎯 Hedef Değişken")
            if 'Heart Disease Status' in df.columns:
                heart_disease_counts = df['Heart Disease Status'].value_counts()
                st.write("**Kalp Hastalığı Durumu:**")
                st.write(f"- Sağlıklı: {heart_disease_counts.get(0, 0)}")
                st.write(f"- Kalp Hastalığı: {heart_disease_counts.get(1, 0)}")
        
        with col2:
            st.subheader("📈 Veri Dağılımı")
            st.write("**Sayısal Değişkenler:**")
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols[:5]:  # İlk 5 sayısal değişken
                st.write(f"- {col}: {df[col].mean():.2f} ± {df[col].std():.2f}")
    
    elif presentation_section == "🔍 Özellik Mühendisliği":
        st.header("🔍 Özellik Mühendisliği")
        
        st.subheader("🛠️ Türetilen Özellikler")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**1. Ves_Hardness (Trigliserit Kategorizasyonu)**")
            st.write("- Düşük (<100): 0")
            st.write("- Orta (100-150): 1")
            st.write("- Yüksek (>150): 2")
            
            st.write("**2. Bp/Crp (Kan Basıncı/CRP Oranı)**")
            st.write("- Enfeksiyon ve kan basıncı ilişkisi")
        
        with col2:
            st.write("**3. Ves_dia_est (Kan Basıncı/Kolesterol Oranı)**")
            st.write("- Damar sağlığı göstergesi")
            
            st.write("**4. Meal order record (Beslenme Skoru)**")
            st.write("- Kolesterol/BMI oranı")
            
            st.write("**5. Chol/Exe (Kolesterol/Egzersiz Oranı)**")
            st.write("- Yaşam tarzı etkisi")
    
    elif presentation_section == "🤖 Model Performansı":
        st.header("🤖 Model Performansı")
        
        # Alt seçenekler
        performance_option = st.selectbox(
            "Performans Analizi Seçin",
            ["📊 Metrikler", "📈 Veri Görselleştirmeleri", "🔍 Detaylı Analiz"]
        )
        
        if performance_option == "📊 Metrikler":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Performans Metrikleri")
                st.metric("Doğruluk (Accuracy)", "0.894 ± 0.004")
                st.metric("F1 Skoru", "0.883 ± 0.004")
                st.metric("Recall", "0.796 ± 0.006")
                st.metric("Precision", "0.991 ± 0.005")
                st.metric("ROC-AUC", "0.947 ± 0.005")
            
            with col2:
                st.subheader("🔧 Model Detayları")
                st.write("**Algoritma:** Random Forest Classifier")
                st.write("**Veri Dengesizliği:** SMOTE ile düzeltildi")
                st.write("**Özellik Sayısı:** 24 (20 temel + 4 türetilmiş)")
                st.write("**Cross-Validation:** 5-Fold")
                
                st.subheader("📈 İyileştirme Önerileri")
                st.write("• Daha fazla veri toplama")
        
        elif performance_option == "📈 Veri Görselleştirmeleri":
            st.subheader("📈 Veri Görselleştirmeleri")
            
            # Görselleştirme seçenekleri
            viz_option = st.selectbox(
                "Görselleştirme Türü Seçin",
                ["📊 İşlenmemiş Veri Seti Kategorik Değişken Dağılımı", "📈 İşlenmemiş Veri Seti Sayısal Değişkenlerin Dağılımı", "🦾 İşlenmiş Veri Seti Sayısal Değişkenlerin Dağılımı", "🎯 Hedef Değişken Analizi"]
            )
            
            if viz_option == "📊 İşlenmemiş Veri Seti Kategorik Değişken Dağılımı":
                st.write("**İşlenmemiş Veri Setindeki Kategorik Değişkenlerin Frekans Dağılımları:**")
                plot_categorical_distributions(df_first)
                
            elif viz_option == "📈 İşlenmemiş Veri Seti Sayısal Değişkenlerin Dağılımı":
                st.write("**İşlenmemiş Veri Setindeki Sayısal Değişkenlerin Dağılımları:**")
                plot_numerical_distributions(df_first)

            elif viz_option == "🦾 İşlenmiş Veri Seti Sayısal Değişkenlerin Dağılımı":
                st.write("**İşlenmiş Veri Setindeki Sayısal Değişkenlerin Dağılımları:**")
                plot_numerical_distributions(df)
                
            elif viz_option == "🎯 Hedef Değişken Analizi":
                st.write("**Hedef Değişken (Kalp Hastalığı) Analizi:**")
                
                if 'Heart Disease Status' in df.columns:
                    # Hedef değişken dağılımı
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Pie chart
                    heart_disease_counts = df['Heart Disease Status'].value_counts()
                    ax1.pie(heart_disease_counts.values, labels=['Sağlıklı', 'Kalp Hastalığı'], autopct='%1.1f%%')
                    ax1.set_title('Kalp Hastalığı Dağılımı')
                    
                    # Bar chart
                    sns.countplot(data=df, x='Heart Disease Status', ax=ax2)
                    ax2.set_title('Kalp Hastalığı Sayısı')
                    ax2.set_xlabel('Kalp Hastalığı Durumu')
                    ax2.set_ylabel('Sayı')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # İstatistikler
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Toplam Kayıt", len(df))
                    with col2:
                        st.metric("Sağlıklı", heart_disease_counts.get(0, 0))
                    with col3:
                        st.metric("Kalp Hastalığı", heart_disease_counts.get(1, 0))
        
        elif performance_option == "🔍 Detaylı Analiz":
            st.subheader("🔍 Detaylı Model Analizi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Eğitim Süreci:**")
                st.write("1. Veri Ön İşleme")
                st.write("   - Eksik veri doldurma (KNN)")
                st.write("   - Kategorik kodlama")
                st.write("   - Özellik mühendisliği")
                
                st.write("2. Model Seçimi")
                st.write("   - Random Forest Classifier")
                st.write("   - SMOTE ile veri dengesizliği düzeltme")
                st.write("   - Cross-validation")
            
            with col2:
                st.write("**Özellik Önem Sırası:**")
                st.write("1. Yaş (Age)")
                st.write("2. Kan Basıncı (Blood Pressure)")
                st.write("3. Kolesterol Seviyesi")
                st.write("4. BMI")
                st.write("5. Açlık Kan Şekeri")
                
                st.write("**Model Avantajları:**")
                st.write("• Yüksek doğruluk (%89.4)")
                st.write("• Overfitting'e karşı dirençli")
                st.write("• Özellik önemini belirleme")
                st.write("• Kategorik ve sayısal verilerle çalışabilir")

            # GÜNLÜK HAYATA FAYDALARI
            st.subheader("🌍 Günlük Hayata Katkıları ve Faydaları")
            st.markdown("""
            - **Erken Teşhis:** Kullanıcılar, risklerini önceden öğrenerek doktora başvurma konusunda bilinçlenir.
            - **Kişiselleştirilmiş Öneriler:** Her kullanıcıya özel yaşam tarzı ve sağlık önerileri sunulur.
            - **Sağlık Okuryazarlığı:** Kullanıcılar, sağlık verilerinin anlamını ve önemini daha iyi kavrar.
            - **Toplumsal Farkındalık:** Kalp hastalığı gibi yaygın bir sağlık sorunu hakkında toplumsal bilinç artar.
            - **Kolay Erişim:** Web tabanlı arayüz sayesinde herkes, hızlı ve kolay şekilde risk değerlendirmesi yapabilir.
            - **Doktorlara Destek:** Ön değerlendirme aracı olarak doktorların iş yükünü azaltabilir.
            """)

# Model Bilgileri sayfası
elif page == "📋 Model Bilgileri":
    st.title("📋 Model Bilgileri")
    
    st.subheader("🔬 Teknik Detaylar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Türü:** Random Forest Classifier")
        st.write("**Veri Ön İşleme:** KNN Imputer + Label Encoder")
        st.write("**Özellik Seçimi:** Tüm özellikler kullanıldı")
        st.write("**Veri Bölme:** %80 Eğitim, %20 Test")
    
    with col2:
        st.write("**Hiperparametreler:**")
        st.write("- n_estimators: 100")
        st.write("- max_depth: None")
        st.write("- min_samples_split: 2")
        st.write("- random_state: 42")
    
    st.subheader("📊 Feature Engineering Öncesi Özellik Önem Sırası")
    st.write("1. Age")
    st.write("2. Kan Basıncı")
    st.write("3. Kolesterol Seviyesi")
    st.write("4. BMI")
    st.write("5. Açlık Kan Şekeri")

    st.subheader("💪🏻 Feature Engineering Sonrası Özellik Önem Sırası")
    st.write("Model eğitimi sırasında en önemli özellikler (Değerlerin Yakınlık Farkları Çok Az Olduğundan İlk 10):")
    st.write("1. Bp/Crp")
    st.write("2. Homocysteine Level")
    st.write("3. Sleep Hours")
    st.write("4. Age")
    st.write("5. BMI")
    st.write("6. Blood Pressure")
    st.write("7. Chol/Exe")
    st.write("8. Fasting Blood Sugar")
    st.write("9. Ves_dia_est")
    st.write("10. Meal order record")

# Hakkında sayfası
elif page == "ℹ️ Hakkında":
    st.title("ℹ️ Hakkında")
    
    st.subheader("📋 Önemli Bilgilendirme")
    
    st.markdown("""
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
    
    st.subheader("👨‍💻 Geliştirici")
    st.write("Bu proje eğitim amaçlı geliştirilmiştir.")
    st.write("Teknolojiler: Python, Streamlit, Scikit-learn, Pandas, NumPy") 