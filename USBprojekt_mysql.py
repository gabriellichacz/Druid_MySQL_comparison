#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############## Import bibliotek ##############
# Kwerendy mysql
import sqlalchemy as db
from sqlalchemy import create_engine
# Klastrowanie 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# SVM
#1
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
#2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

############## Polaczenie z baza danych w MySQL ##############
t_start = time.time() # start mierzenie czasu
engine = create_engine('mysql+pymysql://mysql_admin:mysql_admin@localhost:3306/telecom')
connection = engine.connect()
metadata = db.MetaData()
telecom = db.Table('telecom_users', metadata, autoload=True, autoload_with=engine)

############## Kwerenda ##############
# 1 - Wczytanie konkretnych danych
query_1 = db.select([telecom.columns.Contract, db.func.sum(telecom.columns.TotalCharges).label('SumTotalCharges')]).group_by(telecom.columns.Contract)
ResultProxy_1 = connection.execute(query_1)
ResultSet_1 = ResultProxy_1.fetchall()
df_contracts_sum = pd.DataFrame(ResultSet_1)
df_contracts_sum.columns = ResultSet_1[0].keys()
print(df_contracts_sum)
# 2 - Wczytanie calej tabeli
query_2 = db.select([telecom])
ResultProxy_2 = connection.execute(query_2)
ResultSet_2 = ResultProxy_2.fetchall()
df_data = pd.DataFrame(ResultSet_2)
df_data.columns = ResultSet_2[0].keys()
df_data.tenure = pd.to_numeric(df_data.tenure)
df_data.MonthlyCharges = pd.to_numeric(df_data.MonthlyCharges)
df_data.TotalCharges = pd.to_numeric(df_data.TotalCharges)

# 3 - Wczytanie tylko potrzebnych kolumn
query_3 = db.select([telecom.columns.tenure, telecom.columns.MonthlyCharges])
ResultProxy_3 = connection.execute(query_3)
ResultSet_3 = ResultProxy_3.fetchall()
df_cluster1 = pd.DataFrame(ResultSet_3)
df_cluster1.columns = ResultSet_3[0].keys()
df_cluster1.tenure = pd.to_numeric(df_cluster1.tenure)
df_cluster1.MonthlyCharges = pd.to_numeric(df_cluster1.MonthlyCharges)

############## Rysowanie wykresu dla df_contracts_sum ##############
df_contracts_sum.SumTotalCharges = pd.to_numeric(df_contracts_sum.SumTotalCharges)
df_contracts_sum = df_contracts_sum.sort_values(by=['SumTotalCharges'], ascending=False)
df_contracts_sum.plot(x='Contract', kind='bar')
plt.show()

############## Klasteryzacja df_cluster1 ##############
# elbowscore - wybranie ile klastrow powinno byc
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_cluster1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Liczba klastrow')
plt.ylabel('WCSS')
plt.show()

# klasteryzacja
kmeans = KMeans(4)
kmeans.fit(df_cluster1)
identified_clusters = kmeans.fit_predict(df_cluster1)

data_with_clusters = df_data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['tenure'], data_with_clusters['MonthlyCharges'], c = data_with_clusters['Clusters'], cmap = 'rainbow')
plt.title('Metoda k-srednich')
plt.xlabel('tenure')
plt.ylabel('MonthlyCharges')
plt.show()

# klasteryzacja ze standaryzowaniem
df_cluster1_std = StandardScaler().fit_transform(df_cluster1) # standaryzacja danych

kmeans = KMeans(4)
kmeans.fit(df_cluster1_std)
identified_clusters = kmeans.fit_predict(df_cluster1_std)

data_with_clusters_std = df_data.copy()
data_with_clusters_std['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters_std['tenure'], data_with_clusters_std['MonthlyCharges'], c = data_with_clusters_std['Clusters'], cmap = 'rainbow')
plt.title('Metoda k-srednich - standaryzacja')
plt.xlabel('tenure')
plt.ylabel('MonthlyCharges')
plt.show()

# Oba sposoby klasteryzacji zwrocily podobne wyniki. Klienci podzieleni sa na 4 grupy.
# Grupa 1 - placacy krotko i malo.
# Grupa 2 - placacy dlugo i malo.
# Grupa 3 - placacy krotko i duzo.
# Grupa 4 - placacy dlugo i duzo.
# Dzieki podzialowi firma moze zaplanowac dzialanie i skierowac odpowiednie oferty dla danej grupy klientow.

# Celem firmy w stosunku do grupy 1 powinno byc przekonanie klientow by pozostali jak najdluzej i zdecydowali sie na drozsza usluge lub wiecej uslug.
#   Np. czasowa znizka na drozsza usluge oraz program lojalnosciowy.
# Celem firmy w stosunku do grupy 2 powinno byc przekonanie klientow by zdecydowali sie na drozsza usluge lub wiecej uslug.
#   Np. czasowa znizka na drozsza usluge.
# Celem firmy w stosunku do grupy 3 powinno byc przekonanie klientow by pozostali jak najdluzej. 
#   Np. program lojalnosciowy z wyszczegolnionymi jasno znizkami na przyszlosc
# Celem firmy w stosunku do grupy 4 powinno byc utrzymanie jak najdluzej lojalnych klientow. 
#   Mozna zaoferowac im np. program lojalnosciowy ze znizkami lub nowymi uslugami.

############# SVM - data_with_clusters_std ##############
data_svm = 0;
data_svm = data_with_clusters_std;

# Zamiana Yes na 1 i No na 0
data_svm.Partner = pd.Series(np.where(data_svm.Partner.values == 'Yes', 1, 0), data_svm.index)
data_svm.Dependents = pd.Series(np.where(data_svm.Dependents.values == 'Yes', 1, 0), data_svm.index)
data_svm.PhoneService = pd.Series(np.where(data_svm.PhoneService.values == 'Yes', 1, 0), data_svm.index)
data_svm.PaperlessBilling = pd.Series(np.where(data_svm.PaperlessBilling.values == 'Yes', 1, 0), data_svm.index)
data_svm.Churn = pd.Series(np.where(data_svm.Churn.values == 'Yes', 1, 0), data_svm.index)
data_svm.OnlineSecurity = pd.Series(np.where(data_svm.OnlineSecurity.values == 'Yes', 1, 0), data_svm.index)
data_svm.OnlineBackup = pd.Series(np.where(data_svm.OnlineBackup.values == 'Yes', 1, 0), data_svm.index)
data_svm.DeviceProtection = pd.Series(np.where(data_svm.DeviceProtection.values == 'Yes', 1, 0), data_svm.index)
data_svm.TechSupport = pd.Series(np.where(data_svm.TechSupport.values == 'Yes', 1, 0), data_svm.index)
data_svm.StreamingTV = pd.Series(np.where(data_svm.StreamingTV.values == 'Yes', 1, 0), data_svm.index)
data_svm.StreamingMovies = pd.Series(np.where(data_svm.StreamingMovies.values == 'Yes', 1, 0), data_svm.index)
data_svm.gender = pd.Series(np.where(data_svm.gender.values == 'Female', 1, 0), data_svm.index) # kobieta=1, mezczyzna=0
data_svm.MultipleLines = pd.Series(np.where(data_svm.MultipleLines.values == 'Yes', 1, 0), data_svm.index)
data_svm.InternetService = pd.Series(np.where(data_svm.InternetService.values == 'No', 0, 1), data_svm.index)

# Usuwanie niepotrzebnych kolumn
data_svm = data_svm.drop(['customerID', 'Contract', 'PaymentMethod', 'Churn'], axis = 1)

# Dane
X = data_svm
X = X.drop(['Clusters'], axis = 1)
Y = data_svm.Clusters # klasy

# Parametry dla klasyfikatora
gamma = [0.005, 0.01, 0.05, 0.2, 0.8, 1.5, 2.5, 5, 10, 20, 50]
gamma = np.array(gamma)
C = [1, 10, 100, 1000, 10000, 100000]
C = np.array(C)

# Tablice do zapisywania dokladnosci
Accuracy_CV = np.zeros((10,1))
Accuracy = np.zeros((len(gamma), len(C)))

for i in range(0, len(C)): # C
    for j in range(0, len(gamma)): # gamma
        for k in range(1, 10): # walidacja krzyzowa
            # Podzielenie danych na zbiory do walidacji i uczenia
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)
            
            # Standaryzacja
            sc = StandardScaler()
            sc.fit(X_train)
            X_train = sc.transform(X_train)
            X_test = sc.transform(X_test)
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            
            # Model
            svclassifier = SVC(C = C[i], gamma = gamma[j])
            
            svclassifier.fit(X_train, Y_train)
            y_pred = svclassifier.predict(X_test)
            
            # Sprawdzenie dokladnosci - czyli ile wartosci ze zbioru y_pred jest rowne Y_test
            Accuracy_CV[k] = sum(y_pred == Y_test)/len(Y_test)

        Accuracy[j,i] = np.mean(Accuracy_CV) # wiersze to zmiana gamma, kolumny to zmiana C
        
# na osi X jest gamma, rozne wykresy to C        
for p in range(0, len(C)): # rysuje tyle wykresow ile jest wartosci C 
    plt.plot(Accuracy[:,p], label = C[p]) # kazdy wykres osobno, zeby kazdy mial swoja nazwe
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.xticks(np.arange(len(gamma)), gamma) # prawidlowe oznaczenie osi X
plt.legend(title = 'C')

t_end = time.time() # koniec umierzenia czasu
print("\nWykonanie calego kodu z polaczeniem z MySQL zajelo", t_end - t_start, "sekund")