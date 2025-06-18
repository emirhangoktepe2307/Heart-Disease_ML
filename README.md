# ❤️ Kalp Hastalığı Tahmin Uygulaması

Bu proje, makine öğrenimi kullanarak kalp hastalığı riskini tahmin eden bir Streamlit web uygulamasıdır.

## 🚀 Özellikler

- **Makine Öğrenimi Modeli**: Random Forest Classifier
- **Veri Dengesizliği Düzeltme**: SMOTE algoritması
- **Özellik Mühendisliği**: 4 yeni özellik türetme
- **Kullanıcı Dostu Arayüz**: Streamlit ile modern web arayüzü
- **Risk Seviyesi Analizi**: Detaylı risk değerlendirmesi

## 📊 Model Performansı

- **Doğruluk (Accuracy)**: %71.6
- **F1 Skoru**: 0.137
- **Recall**: 0.116
- **Precision**: 0.166
- **ROC-AUC**: 0.488

## 🛠️ Kurulum

### Gereksinimler

```bash
pip install -r requirements.txt
```

### Gerekli Dosyalar

1. `heart_disease.csv` - Ham veri dosyası
2. `model_pred.py` - Model eğitimi
3. `app.py` - Streamlit uygulaması
4. `requirements.txt` - Python bağımlılıkları

## 🔧 Kullanım

### 1. Model Eğitimi

```bash
python model_pred.py
```

Bu komut:
- Veriyi ön işler
- Modeli eğitir
- `heart_pipeline.joblib` dosyasını oluşturur
- `heart_disease_feature.csv` dosyasını oluşturur

### 2. Streamlit Uygulaması

```bash
streamlit run app.py
```

## 📁 Proje Yapısı

```
Streamlit_ML/
├── app.py                    # Streamlit uygulaması
├── model_pred.py             # Model eğitimi
├── requirements.txt          # Python bağımlılıkları
├── README.md                # Proje dokümantasyonu
├── heart_disease.csv        # Ham veri
├── heart_disease_feature.csv # İşlenmiş veri
└── heart_pipeline.joblib    # Eğitilmiş model
```

## 🎯 Özellikler

### Giriş Parametreleri
- Yaş, Cinsiyet
- Kan Basıncı, Kolesterol
- BMI, Açlık Kan Şekeri
- Trigliserit, CRP, Homosistein
- Yaşam Tarzı Faktörleri (Sigara, Alkol, Egzersiz)
- Aile Geçmişi

### Türetilen Özellikler
- `Ves_Hardness`: Trigliserit kategorizasyonu
- `Bp/Crp`: Kan basıncı/CRP oranı
- `Ves_dia_est`: Kan basıncı/kolesterol oranı
- `Meal order record`: Beslenme skoru
- `Chol/Exe`: Kolesterol/egzersiz oranı

## ⚠️ Önemli Notlar

- Bu uygulama sadece tahmin amaçlıdır
- Tıbbi teşhis aracı değildir
- Sağlık sorunları için mutlaka uzmana başvurun

## 🔗 Streamlit Cloud

Bu uygulama Streamlit Cloud'da deploy edilebilir.

## 📝 Lisans

Bu proje eğitim amaçlıdır.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📞 İletişim

Proje hakkında sorularınız için issue açabilirsiniz. 