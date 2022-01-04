# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 17:53:31 2020

@author: ASUS
"""

from preprocessing import preprocessing
from pembobotan import pembobotan
from pembobotan import InformationGain
from KNN import KNN
from Evaluasi import Evaluasi
import pandas as pd
import numpy as np
import time

start_time = time.time()
dataLatih = pd.read_excel('Dataset/LatihFold4-Copy.xlsx') 
pp = preprocessing(dataLatih['Teks']) 
SimpanHasil = pp.RunAll() 
bobot = pembobotan(SimpanHasil) 
TermF = bobot.TermFrequency() 
print ("\nHasil Term Frequency\n", TermF)
BTF = bobot.BinaryFrequency() 
print ("\nHasil BTF:\n", BTF)
frequency = bobot.DocumentFrequency() 
print ("\nHasil Frequency:\n", frequency)
IG = InformationGain(BTF, frequency, dataLatih['Klasifikasi']) 
TermIG = (IG.InformationGain(0.93))
TFIDF = bobot.TermFrequencyIDF(TermIG) 
IDF = bobot.InverseDocumentFrequency(TermIG) 

# ~~datauji~~
dataUji = pd.read_excel('Dataset/UjiFold4-Copy.xlsx') 
ppUji = preprocessing(dataUji['Teks']) 
SimpanHasilUji = ppUji.RunAll()
def cek(TermLatih, TermUji): 
    HasilCek = [] 
    for i in TermUji: 
        temp = [] 
        for j in i:
            if j in TermLatih:
                temp.append(j) 
        HasilCek.append(temp) 
    return HasilCek 
    HasilCek = [] 
    for i in SimpanHasilUji: 
        if i in TermLatih: 
            HasilCek.append(i)
    return HasilCek

TermUji = cek(TermIG, SimpanHasilUji) 
bobotUji = pembobotan(dataUji['Teks']) #inisialisasi untuk memanggil pembobotan dataUji
TFIDFUji = bobotUji.TermFrequencyIDFUji(TermIG, TermUji, IDF) #manggil TFIDF Uji
####kalo mau perkalimat dimsukin manual
##coba = ["Atika Anggraeni"]
##pp = preprocessing(coba)
##SimpanHasil = pp.RunAll()
#print (TermUji)
k = 10

knn = KNN(TFIDF,TFIDFUji, dataLatih['Klasifikasi'], k)

HasilPelatihan = knn.Pelatihan()
HasilPengujian = knn.Pengujian()
##
#hasilPelatihan = KNN.pelatihan(TFIDF, TFIDFUji)
print ("\nHasil Pelatihan:\n", HasilPelatihan)
for i,j in enumerate(HasilPengujian):
    print ("-{}:{}".format(i+1,j))  
Eval = Evaluasi(HasilPengujian, dataUji["Klasifikasi"]) 
CM = Eval.ConfusionMatrix() 
print (CM)
HasilEvaluasi = Eval.RunAll()
print ("\nK = ", k)
print("\n---%s detik---"%(time.time() - start_time))


