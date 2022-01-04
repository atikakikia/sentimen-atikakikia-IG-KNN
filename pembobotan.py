"""
Created on Wed Sep  2 19:53:19 2020

@author: ASUS
"""
import pandas as pd
import numpy as np
np.seterr(divide = 'ignore')

class pembobotan(object):
    
    def __init__(self, document):
        self.document = document 
        self.kolom = [i+1 for i in range(len(self.document))] 
    
    def TermFrequency(self, indexTerm = []): #sebelum dan sesudah di IG
        term = np.unique(sum(self.document, [])) 
        TF = pd.DataFrame(0, index = term, columns = self.kolom) 
        for i, doc in enumerate(self.document): 
            for term in doc:
                TF.loc[term, i+1] += 1  
        if len(indexTerm) != 0: 
            return TF.loc[indexTerm] 
        return TF
    
    def BinaryFrequency(self):
        term = np.unique(sum(self.document, []))
        BTF = pd.DataFrame(0, index = term, columns = self.kolom)
        for i, doc in enumerate(self.document):
            for term in doc:
                BTF.loc[term, i+1] = 1
        return BTF
    
    def DocumentFrequency(self, indexTerm = []): 
        TF = self.TermFrequency(indexTerm) 
        frequency = (TF>0).values.sum(axis = 1) 
        return frequency
          
    def LogTermFrequency(self, indexTerm = []):
        TF = self.TermFrequency(indexTerm) 
        PerhitunganLogTF = np.where(TF != 0, 1+np.log10(TF), 0)
        LogTF = pd.DataFrame(PerhitunganLogTF,index = TF.index, columns = self.kolom) 
#        print ("\nLog Term Frequency:\n", LogTF)
        return LogTF
        
    def InverseDocumentFrequency(self, indexTerm = []):
        frequency = self.DocumentFrequency(indexTerm) 
        IDF = pd.DataFrame(np.log10(len(self.document)/frequency), index = indexTerm) 
#        print ("\nHasil Inverse Document:\n", IDF)
        return IDF
    
    def TermFrequencyIDF(self, indexTerm = []):
        TF = self.LogTermFrequency(indexTerm) 
        IDF = self.InverseDocumentFrequency(indexTerm) 
        TFIDF = pd.DataFrame(TF.values*IDF.values, index = TF.index, columns = self.kolom) 
#        print ("\nHasil TF-IDF:\n", TFIDF)
        return TFIDF
    
    def TermFrequencyUji(self, TermLatih, TermUji):
        TFUji = pd.DataFrame(0, index = TermLatih, columns = self.kolom) 
        for i,j in zip(TermUji,range(len(self.kolom))): 
            for  doc in i:
                TFUji.loc[doc, j+1] += 1 
#        print ("\nHasil TF Data Uji:\n", TFUji)
        return TFUji
    
    def TermFrequencyIDFUji(self, TermLatih, TermUji, IDF):
        TFUji = self.TermFrequencyUji(TermLatih, TermUji)
        PerhitunganLogTF = np.where(TFUji != 0, 1+np.log10(TFUji), 0) 
        LogTF = pd.DataFrame(PerhitunganLogTF,index = TFUji.index, columns = self.kolom)
        TFIDF = pd.DataFrame(LogTF.values*IDF.values, index = TFUji.index, columns = self.kolom) #perhitungan TF-IDF uji saja tanpa data latih
#        print ("\nHasil TF-IDF Data Uji:\n", TFIDF)
        return TFIDF
    
class InformationGain(object):
    
    def __init__(self, BTF, DF, klasifikasi):
        self.BTF = BTF
        self.DF = DF
        self.klasifikasi = klasifikasi
        
    def InformationGain(self, percent):

        DFGrup = self.klasifikasi.groupby(self.klasifikasi) 
        Kelas = {} 
        for i in np.unique(self.klasifikasi):
            temp = DFGrup.get_group(i).index.tolist()
            Kelas[i] = [j+1 for j in temp] 
        PeluangKelas = {} 
        
        for i in Kelas: 
            PeluangKelas[i] = len(Kelas[i])/len(self.klasifikasi) #P(Pos) dan P(Neg)
#        print ("\nHasil Peluang Setiap Kelas:\n", PeluangKelas)
        
        PTermKelas = pd.DataFrame(0, index = self.BTF.index, columns = Kelas) #P(Pos|t) dan P(Neg|t) 
        for i in Kelas: 
            PTermKelas[i] = self.BTF[Kelas[i]].sum(axis = 1)/len(Kelas[i]) #P(Pos|t)
#        print ("\nPeluang Term Terhadap Kelas:\n", PTermKelas)
        
        NPTermKelas = pd.DataFrame(0, index = self.BTF.index, columns = Kelas) #P(Pos|~t) dan P(Neg|~t) 
        for i in Kelas: 
            NPTermKelas[i] = (len(Kelas[i])-self.BTF[Kelas[i]].sum(axis = 1))/len(Kelas[i]) #P(Pos|~t)
#        print ("\nNegasi Peluang Term Terhadap Kelas:\n", NPTermKelas)

        PerhitunganLog = np.where(PTermKelas != 0, np.log10(PTermKelas),0) #Log P(Pos|t) dan log P(Neg|t)
        PLogTerm = pd.DataFrame(PerhitunganLog, index = PTermKelas.index, columns = PTermKelas.columns)         
#        print ("\nHasil Peluang Log Term :\n", PLogTerm)

        NPerhitunganLog = np.where(NPTermKelas > 0, np.log10(NPTermKelas),0) 
        NPLogTerm = pd.DataFrame(NPerhitunganLog, index = NPTermKelas.index, columns = NPTermKelas.columns)
#        print ("\nNegasi Peluang Log Term:\n", NPLogTerm)

        PeluangTerm = pd.Series(self.DF/sum(self.DF), index = self.BTF.index) 
#        print ("\nHasil Peluang Term:\n", PeluangTerm)
        NPeluangTerm = pd.Series((len(self.BTF.columns)-self.DF)/sum (self.DF), index = self.BTF.index) #P(~t)
#        print ("\nHasil Negasi Peluang Term:\n", NPeluangTerm)
          
        IG1 = 0 
        for i in PeluangKelas.values(): #looping peluang kelas P(Pos) dan P(Neg)
            IG1 += i*np.log10(i)
        IG1 = -IG1 
#        print ("\nHasil Information Gain 1:\n", IG1)
        
        IG2 = PeluangTerm*(PTermKelas * PLogTerm).sum(axis = 1) 
#        print ("\nHasil Information Gain 2:\n", IG2)
        
        IG3 = NPeluangTerm*(NPTermKelas * NPLogTerm).sum(axis = 1)
#        print ("\nHasil Information Gain 3:\n", IG3)
        
        IG = IG1+IG2+IG3
#        print ("\nHasil Total Information Gain:\n", IG)
        IG = IG.sort_values(ascending = False) 
        index_ =round(len(IG)*percent)
        if (len(IG)-1) < index_:
            index_ = len(IG)-1 
        threshold = IG[index_] 
        HasilIG = IG[(IG >= threshold)]
        print ("\nHasil Information Gain Sesuai Threshold:\n", HasilIG)
        return HasilIG.index.tolist()
        
        
        
        
        