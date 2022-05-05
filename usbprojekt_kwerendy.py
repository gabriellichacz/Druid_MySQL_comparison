#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################ Import bibliotek ############################
# Kwerendy druid
from pydruid import *
from pydruid.client import *
from pylab import plt
from pydruid.query import QueryBuilder
from pydruid.utils.postaggregator import *
from pydruid.utils.aggregators import *
from pydruid.utils.filters import *
# Kwerendy mysql
import sqlalchemy as db
from sqlalchemy import create_engine, text
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

############################ Komentarz ############################
"""
    Czas mierzony od poczatku zapytania do zapisania danych w najbardziej podstawowej formie ramki danych pandas.
"""

############################ Mniejsza baza danych ############################

############## Apache Druid ##############
# Polaczenie z baza danych
t_start_druid_1 = time.time() # start mierzenie czasu
query = PyDruid('http://localhost:8888', 'druid/v2/')
t_end_druid_1 = time.time() # koniec mierzenia czasu
print("\nWykonanie polaczenia z baza danych Druid zajelo", t_end_druid_1 - t_start_druid_1, "sekund")

# 1 - Wczytanie konkretnych danych
t_start_druid_2 = time.time() # start mierzenie czasu
contracts_sum = query.topn(
    datasource = "telecom_users", # tabela z danymi
    granularity = "all",
    intervals = "-146136543-09-08T08:23:32.096Z/146140482-04-24T15:36:27.903Z", # duzy interwal czasowy
    dimension = "Contract", # chce wyswietlic dane dla wszystkich wartosci Contract 
                            # (czyli policzone dla wszystkich rodzajow umow)
    filter = Dimension("PhoneService") == "Yes", # wiesze tylko gdy PhoneService = Yes
                                                 # (czyli policzone dla klientow posiadajacych usluge telefonu)
    aggregations = {"TotalCharges_sum": longsum("TotalCharges")}, # suma TotalCharges
                                                    # (czyli sumaryczna kwota zarobiona juz na wybranych osobach)
    metric = "TotalCharges_sum", # metryka do sortowania
    threshold = 10 # wybieram ile chce wierszy
) # najbardziej podstawowa forma czytelnych danych
t_end_druid_2 = time.time() # koniec umierzenia czasu
df_contracts_sum = query.export_pandas() # ramka danych
print("\nKwerenda zajela", t_end_druid_2 - t_start_druid_2, "sekund")
print("\n", df_contracts_sum)

# 2 - Wczytanie calej tabeli
t_start_druid_3 = time.time() # start mierzenie czasu
data = query.scan(
            datasource = "telecom_users",
            granularity = 'all',
            intervals = "-146136543-09-08T08:23:32.096Z/146140482-04-24T15:36:27.903Z"
) # najbardziej podstawowa forma czytelnych danych
t_end_druid_3 = time.time() # koniec mierzenia czasu
df_data = query.export_pandas() # ramka danych
print("\nKwerenda zajela", t_end_druid_3 - t_start_druid_3, "sekund")
print("\n", df_data)

############## MySQL ##############
# Polaczenie z baza danych
t_start_mysql_1 = time.time() # start mierzenie czasu
engine = create_engine('mysql+pymysql://mysql_admin:mysql_admin@localhost:3306/telecom')
connection = engine.connect()
metadata = db.MetaData()
telecom = db.Table('telecom_users', metadata, autoload=True, autoload_with=engine)
t_end_mysql_1 = time.time() # koniec mierzenia czasu
print("\nWykonanie polaczenia z baza danych MySQL zajelo" , t_end_mysql_1 - t_start_mysql_1, "sekund")

# 1 - Wczytanie konkretnych danych
t_start_mysql_2 = time.time() # start mierzenie czasu
query_1 = db.select([telecom.columns.Contract, db.func.sum(telecom.columns.TotalCharges).label('SumTotalCharges')]).group_by(telecom.columns.Contract)
ResultProxy_1 = connection.execute(query_1)
ResultSet_1 = ResultProxy_1.fetchall() # najbardziej podstawowa forma czytelnych danych
t_end_mysql_2 = time.time() # koniec mierzenia czasu
df_contracts_sum_1 = pd.DataFrame(ResultSet_1) # ramka danych
print("\nKwerenda zajela" , t_end_mysql_2 - t_start_mysql_2, "sekund")
df_contracts_sum_1.columns = ResultSet_1[0].keys()
print("\n", df_contracts_sum_1)

# 2 - Wczytanie calej tabeli
t_start_mysql_3 = time.time() # start mierzenie czasu
query_2 = db.select([telecom])
ResultProxy_2 = connection.execute(query_2)
ResultSet_2 = ResultProxy_2.fetchall() # najbardziej podstawowa forma czytelnych danych
t_end_mysql_3 = time.time() # koniec mierzenia czasu
df_data_1 = pd.DataFrame(ResultSet_2) # ramka danych
print("\nKwerenda zajela", t_end_mysql_3 - t_start_mysql_3, "sekund")
df_data_1.columns = ResultSet_2[0].keys()
df_data_1.tenure = pd.to_numeric(df_data_1.tenure)
df_data_1.MonthlyCharges = pd.to_numeric(df_data_1.MonthlyCharges)
df_data_1.TotalCharges = pd.to_numeric(df_data_1.TotalCharges)
print("\n", df_data_1)

############################ Wieksza baza danych ############################

############## Apache Druid ##############
# Polaczenie z baza danych
t_start_druid_4 = time.time() # start mierzenie czasu
query = PyDruid('http://localhost:8888', 'druid/v2/')
t_end_druid_4 = time.time() # koniec mierzenia czasu
print("\nWykonanie polaczenia z baza danych Druid zajelo", t_end_druid_4 - t_start_druid_4, "sekund")

# Wczytanie danych - filtr
t_start_druid_5 = time.time() # start mierzenie czasu
data = query.scan(
            datasource = "data",
            granularity = 'all',
            intervals = "-146136543-09-08T08:23:32.096Z/146140482-04-24T15:36:27.903Z",
            filter = Dimension("gender") == "male"
) # najbardziej podstawowa forma czytelnych danych
t_end_druid_5 = time.time() # koniec mierzenia czasu
print("\nKwerenda zajela", t_end_druid_5 - t_start_druid_5, "sekund")
print(data)
#df_data2 = query.export_pandas() # ramka danych
#print("\n", df_data2)

# Wczytanie danych - filtr2
t_start_druid_7 = time.time() # start mierzenie czasu
mobile_count = query.topn(
    datasource = "data", # tabela z danymi
    granularity = "all",
    intervals = "-146136543-09-08T08:23:32.096Z/146140482-04-24T15:36:27.903Z", # duzy interwal czasowy
    dimension = "mobile_type", # chce wyswietlic dane dla wszystkich wartosci mobile_type 
                            # (czyli policzone dla wszystkich rodzajow umow)
    filter = Dimension("system_status") == "ACTIVE", # wiesze tylko gdy system_status = ACTIVE
                                                 # (czyli policzone dla klientow posiadajacych usluge telefonu)
    aggregations = {"mobile_type_sum": count("mobile_type")}, # policzona ilosc mobile_type
                                                    # (czyli sumaryczna kwota zarobiona juz na wybranych osobach)
    metric = "mobile_type_sum", # metryka do sortowania
    threshold = 10 # wybieram ile chce wierszy
) # najbardziej podstawowa forma czytelnych danych
t_end_druid_7 = time.time() # koniec mierzenia czasu
print("\nKwerenda zajela", t_end_druid_7 - t_start_druid_7, "sekund")
print(mobile_count)
mobile_count = query.export_pandas() # ramka danych
print("\n", mobile_count)

# Wczytanie calej tabeli
t_start_druid_6 = time.time() # start mierzenie czasu
data2 = query.scan(
            datasource = "data",
            granularity = 'all',
            intervals = "-146136543-09-08T08:23:32.096Z/146140482-04-24T15:36:27.903Z"
) # najbardziej podstawowa forma czytelnych danych
t_end_druid_6 = time.time() # koniec mierzenia czasu
print("\nKwerenda zajela", t_end_druid_6 - t_start_druid_6, "sekund")
print(data2)
#df_data2 = query.export_pandas() # ramka danych
#print("\n", df_data2)

############## MySQL ##############
# Polaczenie z baza danych -- wielkosc nie ma znaczenia
t_start_mysql_4 = time.time() # start mierzenie czasu
engine = create_engine('mysql+pymysql://mysql_admin:mysql_admin@localhost:3306/telecom_big')
connection = engine.connect()
metadata = db.MetaData()
telecom_big = db.Table('crm1', metadata, autoload=True, autoload_with=engine)
t_end_mysql_4 = time.time() # koniec mierzenia czasu
print("\nWykonanie polaczenia z baza danych MySQL zajelo" , t_end_mysql_4 - t_start_mysql_4, "sekund")

# Wczytanie danych - filtr
t_start_mysql_6 = time.time() # start mierzenie czasu
sql_text = text("SELECT * FROM crm1 WHERE gender LIKE 'male'")
result = connection.execute(sql_text) # najbardziej podstawowa forma czytelnych danych
t_end_mysql_6 = time.time() # koniec mierzenia czasu
print("\nKwerenda zajela", t_end_mysql_6 - t_start_mysql_6, "sekund")
df_data_3 = pd.DataFrame(result) # ramka danych
#print("\n", df_data_3)

# Wczytanie danych - filtr2
t_start_mysql_7 = time.time() # start mierzenie czasu
sql_text = text("SELECT count(mobile_type), mobile_type FROM crm1 WHERE system_status LIKE 'ACTIVE' GROUP BY mobile_type")
result = connection.execute(sql_text) # najbardziej podstawowa forma czytelnych danych
t_end_mysql_7 = time.time() # koniec mierzenia czasu
print("\nKwerenda zajela", t_end_mysql_7 - t_start_mysql_7, "sekund")
df_data_4 = pd.DataFrame(result) # ramka danych
#print("\n", df_data_3)

# Wczytanie calej tabeli danych
t_start_mysql_5 = time.time() # start mierzenie czasu
sql_text = text("SELECT * FROM crm1")
result = connection.execute(sql_text) # najbardziej podstawowa forma czytelnych danych
t_end_mysql_5 = time.time() # koniec mierzenia czasu
print("\nKwerenda zajela", t_end_mysql_5 - t_start_mysql_5, "sekund")
df_data_2 = pd.DataFrame(result) # ramka danych
for row in result:
    print(row)
#print("\n", df_data_2)
