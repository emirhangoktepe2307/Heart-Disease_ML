import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import joblib, os

# Veri yükleme
path = "/Users/emirhangoktepe/Desktop/Streamlit_ML/heart_disease.csv"
df = pd.read_csv(path)

# Görselleştirme fonksiyonları
def plot_categorical_distributions(df):
    cat_cols = df.select_dtypes("object").columns
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f"{col} Frekans Dağılımı")
        plt.tight_layout()
        plt.show()

def plot_numerical_distributions(df):
    num_cols = df.select_dtypes(include=["number"]).columns
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
        plt.show()

# Veri ön işleme
def preprocess_data(df):
    df = df.copy()
    
    # Eksik veri doldurma
    numeric_columns = df.select_dtypes(include=['number']).columns
    categoric_columns = df.select_dtypes(include=['object']).columns
    
    imputer_knn = KNNImputer(n_neighbors=5)
    imputer_most_frequent = SimpleImputer(strategy='most_frequent')
    
    for column in numeric_columns:
        df[column] = imputer_knn.fit_transform(df[[column]])
    
    for column in categoric_columns:
        df[column] = imputer_most_frequent.fit_transform(df[[column]]).ravel()
    
    # Kategorik değişkenleri dönüştürme
    df["Exercise Habits"] = df["Exercise Habits"].map({"High": 1, "Low": 3, "Medium": 2}).astype(int)
    
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    
    return df

# Trigliserit kategorizasyonu
def categorize_triglyceride(level):
    if pd.isna(level):
        return np.nan
    elif level < 100:
        return 0
    elif 100 <= level < 150:
        return 1
    else:
        return 2

# Özellik mühendisliği
def add_ratios(X):
    X = X.copy()
    X["Ves_Hardness"] = X["Triglyceride Level"].apply(categorize_triglyceride)
    X["Bp/Crp"] = X["CRP Level"] / X["Blood Pressure"]
    X["Ves_dia_est"] = X["Blood Pressure"] / X["Cholesterol Level"]
    X["Meal order record"] = X["Cholesterol Level"] / X["BMI"]
    X["Chol/Exe"] = X["Cholesterol Level"] / X["Exercise Habits"]
    return X

# Ana işlem
def main():
    # Veri ön işleme
    df_processed = preprocess_data(df)
    
    # Görselleştirme
    plot_categorical_distributions(df_processed)
    plot_numerical_distributions(df_processed)
    
    # Veri ayrımı
    X = df_processed.drop("Heart Disease Status", axis=1)
    y = df_processed["Heart Disease Status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # SMOTE uygulama
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Pipeline oluşturma
    ratio_tf = FunctionTransformer(add_ratios, validate=False)
    ratio_tf.set_output(transform="pandas")
    
    pipe = Pipeline([
        ("ratios", ratio_tf),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])

    df.to_csv("/Users/emirhangoktepe/Desktop/Streamlit_ML/heart_disease_feature.csv", index=False)
    
    # Model eğitimi
    pipe.fit(X_train_resampled, y_train_resampled)
    
    # Model kaydetme
    save_path = "/Users/emirhangoktepe/Desktop/Streamlit_ML/heart_pipeline.joblib"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipe, save_path)
    print(f"✅ Pipeline başarıyla kaydedildi → {save_path}")
      
    # Model değerlendirme
    y_pred = pipe.predict(X_test)
    print("\nModel Performans Metrikleri:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.3f}")

if __name__ == "__main__":
    main()

"""
=== 5‑Fold CV Sonuçları ===
            Model Accuracy (mean±std) F1 (mean±std) Recall (mean±std) Precision (mean±std) ROC‑AUC (mean±std)
    Random Forest       0.894 ± 0.004 0.883 ± 0.004     0.796 ± 0.006        0.991 ± 0.005      0.947 ± 0.005
Gradient Boosting       0.867 ± 0.005 0.847 ± 0.007     0.734 ± 0.010        1.000 ± 0.000      0.881 ± 0.008
              SVM       0.817 ± 0.005 0.811 ± 0.007     0.788 ± 0.015        0.836 ± 0.005      0.890 ± 0.004
    Decision Tree       0.777 ± 0.004 0.782 ± 0.003     0.801 ± 0.008        0.764 ± 0.007      0.777 ± 0.004
      Naive Bayes       0.610 ± 0.010 0.626 ± 0.009     0.654 ± 0.011        0.601 ± 0.010      0.652 ± 0.014

    ~ Colab ortamında model karşılaştırması yaptım ve bu modele en uygun eğitim şeklinin Random Forest Classifier olduğuna karar verdim    
"""