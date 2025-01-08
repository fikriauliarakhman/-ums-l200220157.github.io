# import tarfile
# from itertools import islice
# from metaflow import S3
# 
# def load_yelp_riviews(num_docs):  #A
    # with S3() as S3: #B
        # res = s3.get('s3://fast-ai-nlp/yelp_review_full_csv.tgz') #B
        # with tarfile.open(res.path) as tar : #C
            # datafile = tar.extractfile('yelp_review_full_csv/train.csv') #C
            # return list (islice(datafile, num_docs)) #D
        # 
        # def make_matrix(docs, binary=False): #E
            # from sklearn.feature_extraction.text import CountVectorizer
            # vec = CountVectorizer(min_df=10, max_df=0.1, binary=binary) #F
            # mtx = vec.fit_transform(docs) #F
            # cols = [None] * len(vec.vocabulary_) #G
            # for word, idx in vec.vocabulary_.items(): #G
                # cols[idx] = word #G
            # return mtx, cols
        
#A Fungsi yang memuat kumpulan data dan mengekstrak daftar dokumen darinya
#B Gunakan klien S3 bawaan Metaflow untuk memuat kumpulan data Yelp yang tersedia untuk umum
#C Ekstrak file data dari paket tar
#D Kembalikan baris num_docs pertama dari file
#E Fungsi yang mengubah daftar dokumen menjadi matriks sekumpulan kata
#F CountVectorizer membuat matriks
#G Buat label kolom daftar

import csv
from itertools import islice

def load_whatsapp_data(num_docs):  # Ubah nama fungsi agar sesuai konteks
    """
    Load data from a CSV file containing WhatsApp group messages.
    """
    with open('cleaned_data.csv', 'r', encoding='utf-8') as csvfile:  # Pastikan file CSV sesuai
        reader = csv.reader(csvfile)
        # Skip header and read a limited number of rows
        return list(islice(reader, num_docs))

def make_matrix(docs, binary=False):
    """
    Create a document-term matrix using CountVectorizer.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(min_df=10, max_df=0.1, binary=binary)
    mtx = vec.fit_transform(docs)
    cols = [None] * len(vec.vocabulary_)
    for word, idx in vec.vocabulary_.items():
        cols[idx] = word
    return mtx, cols

# load_whatsapp_data: Mengganti akses ke S3 dengan membaca file lokal (cleaned_data.csv).
# Parameter num_docs: Membatasi jumlah data yang diproses untuk menghindari overload.
# make_matrix: Tetap digunakan untuk menghasilkan matriks dokumen-termo (document-term matrix).