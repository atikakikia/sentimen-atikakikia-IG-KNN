# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:12:51 2020

@author: ASUS
"""
#import pandas as pd

#data = pd.read_excel('E:/Kuliah/Semester 7/Skripsi/Skripsi/AtikaSpyder/Dataset/ManualisasiDataLatih.xlsx')
#print (data['Teks'])
import re

class preprocessing(object):
    def __init__(self, text): #konstraktor, method yg dijalankan terlebih dahulu
        self.text = text
        
    def RunAll(self):
        HasilDataCleaning = self.data_cleaning(self.text)
        HasilCaseFolding = self.case_folding(HasilDataCleaning)
        HasilTokenization = self.Tokenization(HasilCaseFolding)
        HasilFiltering = self.Filtering(HasilTokenization)
        HasilStemming = self.Stemming(HasilFiltering)
        return HasilStemming
    
    def data_cleaning(self, text): 
        HasilDataCleaning = []
        for kalimat in text:
            temp = re.findall(r'\b[A-Za-z]{2,}',kalimat) 
            HasilDataCleaning.append(" ".join(temp)) 
        print ("Hasil Data Cleaning :\n", HasilDataCleaning)
        return HasilDataCleaning
    
    def case_folding(self, text):
        HasilCaseFolding = []
        for kalimat in text:
            temp = kalimat.lower()
            HasilCaseFolding.append(temp)
        print ("\nHasil Case Folding :\n", HasilCaseFolding)
        return HasilCaseFolding
    
    def Tokenization(self, text):
        HasilTokenization = []
        for kalimat in text:
            HasilTokenization.append(kalimat.split(' ')) 
        print ("\nHasil Tokenization :\n", HasilTokenization)
        return HasilTokenization
    
    def Filtering(self, text):
        a = open('tala.csv', 'r')
        stopword = a.read()
        HasilFiltering = []
        for kalimat in text: 
            temp = []
            for token in kalimat: 
                if token not in stopword: 
                    temp.append(token) 
            HasilFiltering.append(temp)
        print ("\nHasil Filtering :\n", HasilFiltering)
        return HasilFiltering
    
    def Stemming(self, text):
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
        factory = StemmerFactory() 
        stemmer = factory.create_stemmer() 
        HasilStemming = []
        for kalimat in text:
            temp = []
            for token in kalimat:
                katadasar = stemmer.stem(token)
                temp.append(katadasar)
            HasilStemming.append(temp)
        print ("\nHasil Stemming :\n", HasilStemming)
        return HasilStemming