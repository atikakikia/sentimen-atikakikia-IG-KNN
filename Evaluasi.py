# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 21:23:34 2020

@author: ASUS
"""
import numpy as np
import pandas as pd

class Evaluasi:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test 
        self.y_pred = y_pred 
        self.Kelas = ["Positif" , "Negatif"] #set kelas positif dan negatif
    
    def RunAll(self):
        Matrix = self.ConfusionMatrix()
#        HasilPrecision = self.precision()
#        HasilRecall = self.recall()
        HasilFmeasure = self.fmeasure()
        HasilAkurasi = self.akurasi()
        return Matrix, HasilFmeasure, HasilAkurasi
    
    def ConfusionMatrix(self):
        Matrix = pd.DataFrame(0,columns = self.Kelas, index = self.Kelas) 
        for i, j in zip(self.y_test, self.y_pred): 
            Matrix.loc[i,j] += 1 
#        print (Matrix)
        return Matrix
    
    def precision(self):
        cm = self.ConfusionMatrix()
        true_positive = cm.loc['Positif','Positif']
        false_positive = cm.loc['Negatif','Positif']
        HasilPrecision = (true_positive/(true_positive+false_positive))*100
        print ("\nHasil Precision :", HasilPrecision)
        return HasilPrecision
    
    def recall(self):
        cm = self.ConfusionMatrix()
        true_positive = cm.loc['Positif','Positif']
        false_negative = cm.loc['Positif','Negatif']
        HasilRecall = (true_positive/(true_positive+false_negative))*100
        print ("\nHasil Recall :", HasilRecall)
        return HasilRecall
#    
    def fmeasure(self):
        HasilPrecision = self.precision()
        HasilRecall = self.recall()
        HasilFmeasure = (2*(HasilPrecision * HasilRecall)/(HasilPrecision + HasilRecall))
        print ("\nHasil F-Measure :", HasilFmeasure)
        return HasilFmeasure
#        
    def akurasi(self):
        cm = self.ConfusionMatrix()
        true_negative = cm.loc['Negatif','Negatif']
        true_positive = cm.loc['Positif','Positif']
        false_negative = cm.loc['Positif','Negatif']
        false_positive = cm.loc['Negatif','Positif']
        HasilAkurasi = (true_positive+true_negative)/(true_negative+false_positive+false_negative+true_positive)*100
        print ("\nHasil Akurasi :", HasilAkurasi)
        return HasilAkurasi