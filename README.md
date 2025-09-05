# ScikitLearn Diabetes Prediction

Bu proje, **diabetes.csv** veri seti kullanılarak yapılan bir sınıflandırma deneyidir. Amaç, hastaların diyabet olup olmadığını tahmin eden bir makine öğrenimi modeli geliştirmektir. Tüm süreç **Python** ve **scikit-learn** kütüphanesi ile gerçekleştirilmiştir.

## Veri Seti

- Kaynak: Kaggle, Pima Indians Diabetes Dataset
- Özellikler:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- Hedef değişken: `Outcome` (0: diyabet yok, 1: diyabet var)

## Veri Ön İşleme

- Eksik değerler `median` ile dolduruldu.
- Sayısal veriler `StandardScaler` ile ölçeklendirildi.
- Kategorik değişkenler (yoksa) sayısal formata çevrildi.

## Modelleme

Denenen sınıflandırma algoritmaları:

- Logistic Regression (LR)
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Gaussian Naive Bayes (GNB)
- Decision Tree Classifier (DTC)
- Random Forest Classifier (RFC)

### Model Seçimi

Modellerin performansları **Accuracy, F1-score ve ROC-AUC** metrikleri ile karşılaştırıldı:

| Model | Accuracy | F1-score | ROC-AUC |
|-------|----------|----------|---------|
| LR    | 0.78     | 0.62     | 0.84    |
| KNN   | 0.75     | 0.58     | 0.76    |
| SVC   | 0.77     | 0.58     | 0.82    |
| GNB   | 0.75     | 0.57     | 0.79    |
| DTC   | 0.69     | 0.53     | 0.65    |
| RFC   | 0.75     | 0.55     | 0.78    |

**Sonuç:** Logistic Regression, dengeli bir performans gösterdiği için seçildi ve optimize edildi.

### Yeni Veri Tahmini

Optimize edilen LR modeli ile yeni hasta verileri tahmin edilebilir. Örnek:

```python
yeni_veri = [[2, 120, 70, 20, 85, 28.0, 0.45, 35]]
yeni_veri_scaled = sc.transform(yeni_veri)
tahmin = optimized_lr.predict(yeni_veri_scaled)
olasılık = optimized_lr.predict_proba(yeni_veri_scaled)
