# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:48:49 2020

@author: ASUS
"""
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore') 
#mengingatkan warning untuk tidak muncul

class KNN(object):
    def __init__(self, tfidf, tfidfuji, klasifikasi, k): 
        self.tfidf = tfidf
        self.tfidfuji = tfidfuji
        self.klasifikasi = klasifikasi
        self.k = k

    def Pelatihan(self):
        TFIDFPangkat = np.square(self.tfidf,dtype='float64')
        TFIDFSum = TFIDFPangkat.sum(axis = 0)
        TFIDFSqrt = np.sqrt(TFIDFSum)
        self.HasilPelatihan = TFIDFSqrt
        return TFIDFSqrt
    
    def Pengujian(self):
        TFIDFUjiPangkat = self.tfidfuji.pow(2)
        TFIDFUjiSum = TFIDFUjiPangkat.sum(axis = 0, skipna = False)
        TFIDFUjiSqrt = np.sqrt(TFIDFUjiSum)
        Latih = self.tfidf
        Uji = self.tfidfuji
        Prediksi = []
        for i in Uji.columns: 
            PerkalianTFIDFLatihUji = pd.DataFrame(Latih.transpose().values * Uji[i].values) 
            PerkalianTFIDFLatihUji = PerkalianTFIDFLatihUji.transpose()
            SumPerkalian = PerkalianTFIDFLatihUji.sum(axis = 0, skipna = False)
            HasilCosSim = self.CosSim(SumPerkalian, TFIDFUjiSqrt[i])
            Sorting = HasilCosSim.sort_values('Hasil CosSim', axis = 0, ascending = False)
            PenentuanK = Sorting.head(self.k)
            temp = PenentuanK['Klasifikasi'].value_counts().idxmax() 
            Prediksi.append(temp) 
#            print ("\nHasil Sorting pada Query {}:\n{}".format(i,Sorting))
#            print('\nJumlah data yang diambil sebanyak K pada Query {}:\n{}'.format(i,PenentuanK))
        return Prediksi
    
    def CosSim(self, SumPerkalian, TFIDFUjiSqrt):
        HasilCosSim = []
        for i,j in zip(SumPerkalian, self.HasilPelatihan): 
            temp = np.divide(i,(j*TFIDFUjiSqrt)) 
            HasilCosSim.append(temp)
        HasilCosSim = pd.DataFrame(HasilCosSim, columns = ['Hasil CosSim'])
        HasilCosSim = HasilCosSim.fillna(0) 
        HasilCosSim['Klasifikasi'] = self.klasifikasi
        HasilCosSim.index = self.tfidf.columns
        return HasilCosSim

