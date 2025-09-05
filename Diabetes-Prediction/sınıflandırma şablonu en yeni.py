# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:16:03 2020

@author: sadievrenseker
"""


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('diabetes.csv')

#encoder: Kategorik -- Numeric(Eksik veirleri doldurma)

from sklearn.impute import SimpleImputer

# Eksik değerleri dolduracağımız sütunlar (0 → NaN olarak değiştirdiğimiz sütunlar)
cols_to_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
veriler[cols_to_impute] = imputer.fit_transform(veriler[cols_to_impute])


x = veriler[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]].values
y = veriler["Outcome"].values



#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#3. SINIFLANDIRMA MODELLERİ

# Buradan itibaren sınıflandırma algoritması başlar

#1 Logistic Regression

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0) #eğitim
logr.fit(X_train, y_train) 
y_pred = logr.predict(X_test) #tahmin


#karmaşıklık matrisi = modelin doğru ve yanlış tahminlerini gösterir
cm = confusion_matrix(y_test,y_pred)
print('LR')
print(cm)


#2.KNN algoritması(k-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('KNN')
print(cm)


#3. SVC(SVM Classifier)
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)


#4. Navie Bayes

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

#5.Decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

#6. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)


    
# 7. ROC EĞRİSİ , TPR, FPR değerleri 

from sklearn import metrics

y_proba = rfc.predict_proba(X_test)

# 1 (Survived) sınıfının olasılıklarını al
fpr, tpr, thold = metrics.roc_curve(y_test, y_proba[:,1], pos_label=1)

print(fpr)
print(tpr)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Modellerin listesi
models = {
    "LR": logr,
    "KNN": knn,
    "SVC": svc,
    "GNB": gnb,
    "DTC": dtc,
    "RFC": rfc
}

# Sonuçları saklamak için liste
results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    
    # Probability veya decision function ile ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        y_proba = model.decision_function(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    results.append([name, acc, f1, roc])
    
    # Confusion matrix'i yazdır
    print(name)
    print(confusion_matrix(y_test, y_pred))
    
# Tabloda göster
import pandas as pd
results_df = pd.DataFrame(results, columns=["Model","Accuracy","F1-score","ROC-AUC"])
print("\nTüm modellerin performansı:")
print(results_df)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Logistic Regression optimizasyonu
# class_weight='balanced' → azınlık sınıfı daha önemli kabul eder
# C parametresi → regularization gücünü ayarlar (daha büyük C → daha az regularization)
optimized_lr = LogisticRegression(C=1.5, class_weight='balanced', solver='liblinear', random_state=0)

# Eğit
optimized_lr.fit(X_train, y_train)

# Tahmin
y_pred = optimized_lr.predict(X_test)
y_proba = optimized_lr.predict_proba(X_test)[:,1]

# Metrikler
print("Optimized LR Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

y_pred = optimized_lr.predict(X_test)           # Sınıf tahmini (0 veya 1)
y_proba = optimized_lr.predict_proba(X_test)[:,1]  # Olasılık tahmini (0–1 arası)

# Diyelim ki yeni bir hasta verisi geldi (8 feature sırasına uygun)
yeni_veri = [[2, 120, 70, 20, 85, 28.0, 0.45, 35]]  

# Önce scale et (daha önce fit edilen sc kullanılarak!)
yeni_veri_scaled = sc.transform(yeni_veri)

# Tahmin et
print(optimized_lr.predict(yeni_veri_scaled))       # 0 veya 1
print(optimized_lr.predict_proba(yeni_veri_scaled)) # [prob_0, prob_1]














