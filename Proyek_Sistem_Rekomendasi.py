# -*- coding: utf-8 -*-
"""Project_Recommendation_System.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/171nkp_V1ZBUwehi50l2tmUxZ_C6QEBNy

<h1> <b>Sistem Rekomendasi Film<b> <h1>

# Import Library

Mengimport library yang dibutuhkan
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

"""# Data Loading

Mengunduh data dari sumber https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset kemudian diunggah melalui *google drive*. Mounted drive ke colab lalu unzip file dan terkahir memuat data ke dalam bentuk dataframe.

variabel yang ada pada dataset:

* books : merupakan daftar buku tersebut.
* ratings : merupakan daftar penilaian yang diberikan pengguna terhadap buku.
* users : merupakan daftar pengguna.

"""

from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/MyDrive/Dataset/Book recomendation.zip"

books = pd.read_csv('/content/Books.csv')
ratings = pd.read_csv('/content/Ratings.csv')
users = pd.read_csv('/content/Users.csv')

"""# EDA - Analisis Univariate
Analisis univariate merupakan proses untuk mengeksplorasi dan menjelaskan setiap variabel dalam kumpulan data secara terpisah.

## Variabel books
"""

books.info()

"""Dapat dilihat bahwa :

*   Terdapat 271360 data dalam books
*   Terdapat 8 buah kolom bertipe objek yaitu ISBN, Book-Title, Book-Author,Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M dan Image-URL-L.


"""

print('Jumlah data buku : ', len(books.ISBN.unique()))

"""Dapat dilihat bahwa jumlah data buku berdasarkan ISBN sebanyak 271360

## Variabel users
"""

users.info()

"""Dapat dilihat bahwa :

*   Terdapat 278858 data dalam users
*   Terdapat 1 buah kolom bertipe int64 yaitu User-ID.
*   Terdapat 1 buah kolom bertipe object yaitu Location.
*   Terdapat 1 buah kolom bertipe float64 yaitu Age.

"""

print('Jumlah data pengguna : ', len(users['User-ID'].unique()))

"""## Variabel ratings"""

ratings.info()

"""Dapat dilihat bahwa :

*   Terdapat 1149780 data dalam ratings.
*   Terdapat 2 buah kolom bertipe int64 yaitu User-ID dan Book-rating.
*   Terdapat 1 buah kolom bertipe object yaitu ISBN.

Karena data terlalu banyak, maka data yang akan digunakan hanya 30000 data saja
"""

# mengambil data sebanyak 30000
ratings = ratings.iloc[:30000,:]

# cek bentuk data
ratings.shape

print('Jumlah data ratings dari user : ', len(ratings['User-ID'].unique()))
print('Jumlah data ratings dari buku : ', len(ratings.ISBN.unique()))

ratings.describe()

"""Dapat kita lihat dari nilai max dan min bahwa nilai rating terbesar yaitu 10 dan nilai rating terkecil yaitu 0

# Content Based Filtering

## Data Preprocessing

### Menggabungkan Book
"""

import numpy as np

# Menggabungkan seluruh ISBN pada kategori books
books_all = np.concatenate((
    books.ISBN.unique(),
    ratings.ISBN.unique(),
))

# Mengurutkan data dan menghapus data yang sama
books_all = np.sort(np.unique(books_all))

print('Jumlah seluruh data buku berdasarkan ISBN: ', len(books_all))

"""### Menggabungkan User"""

# Menggabungkan seluruh User-ID
user_all = np.concatenate((
    ratings['User-ID'].unique(),
    users['User-ID'].unique()
))

# Menghapus data yang sama kemudian mengurutkannya
user_all = np.sort(np.unique(user_all))

print('Jumlah seluruh user: ', len(user_all))

"""### Menggabungkan seluruh data dengan fitur nama buku"""

# Definisikan dataframe rating ke dalam variabel all_books_rate
all_books_rate = ratings
all_books_rate

all_books = pd.merge(all_books_rate, books[['ISBN','Book-Title','Book-Author','Year-Of-Publication','Publisher']], on='ISBN', how='left')
all_books

"""## Data Preparation

### Menangani missing value

Melakukan pengecekan terlebih dahulu apakah didalam dataset terdapat missing value dengan kode berikut :
"""

# cek missing value
all_books.isnull().sum()

"""Terdapat 41772 missing value terhadap fitur Book-Title, Book-Author,Year-Of-Publication, dan Publisher. Karena tidak bisa mengidentifikasi nama bukunya oleh karena itu akan di drop fitur tag menggunakan dropna"""

all_books_clean = all_books.dropna()
all_books_clean

# cek ulang missing value
all_books_clean.isnull().sum()

"""Missing value sudah tidak ada

### Mengurutkan Data

Mengurutkan data secara ascending
"""

# mengurutkan film berdasarkan ISBN ke dalam variabel fix_books
fix_books = all_books_clean.sort_values('ISBN', ascending=True)
fix_books

# cek jumlah fix books
len(fix_books.ISBN.unique())

# Membuat variabel preparation yang berisi dataframe fix_book kemudian mengurutkan berdasarkan ISBN
preparation = fix_books
preparation.sort_values('ISBN', ascending=True)

"""### Menangani data duplikat

Menghapus data yang duplikat dengan fungsi drop_duplicates(). Dalam hal ini, membuang data duplikat pada kolom ‘ISBN’.
"""

# Membuang data duplikat pada variabel preparation
preparation = preparation.drop_duplicates('ISBN')
preparation

"""### Konversi data menjadi list

Melakukan konversi data series menjadi list. Dalam hal ini, menggunakan fungsi tolist() dari library numpy.
"""

# Mengonversi data series ISBN menjadi dalam bentuk list
book_id = preparation['ISBN'].tolist()

# Mengonversi data series ‘Book-Title’ menjadi dalam bentuk list
book_name = preparation['Book-Title'].tolist()

# Mengonversi data series ‘Book-Author’ menjadi dalam bentuk list
book_author = preparation['Book-Author'].tolist()

book_publish = preparation['Publisher'].tolist()

print(len(book_id))
print(len(book_name))
print(len(book_author))
print(len(book_publish))

"""### Membuat Dictionary
Membuat dictionary untuk menentukan pasangan key-value pada data book_id, book_name, book_author, dan book_publish yang telah disiapkan sebelumnya
"""

# Membuat dictionary untuk data book_id, book_name, book_author, dan book_publish
book_new = pd.DataFrame({
    'id': book_id,
    'book_name':book_name,
    'author': book_author,
    'publisher': book_publish
})
book_new

"""## Modelling

### TF-IDF Vectorizer
Digunakan untuk menemukan representasi fitur penting dari setiap penulis buku.
Menggunakan fungsi tfidfvectorizer() dari library sklearn.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi TfidfVectorizer
tfid = TfidfVectorizer()

# Melakukan perhitungan idf pada data author
tfid.fit(book_new['author'])

# Mapping array dari fitur index integer ke fitur nama
tfid.get_feature_names_out()

# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tfid.fit_transform(book_new['author'])

# Melihat ukuran matrix tfidf
tfidf_matrix.shape

"""Dapat dilihat matriks yang dimiliki berukuran (20033, 8832). Nilai 20033 merupakan ukuran data dan 8832 merupakan matrik penulis buku"""

# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()

# Membuat dataframe untuk melihat tf-idf matrix
# Kolom diisi dengan author
# Baris diisi dengan nama buku

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfid.get_feature_names_out(),
    index=book_new.book_name
).sample(10, axis=1,replace=True).sample(10, axis=0,replace=True)

"""### Cosine Similarity
Menghitung derajat kesamaan (similarity degree) antar buku dengan teknik cosine similarity. Dengan menggunakan fungsi cosine_similarity dari library sklearn.
"""

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama buku
cosine_sim_df = pd.DataFrame(cosine_sim, index=book_new['book_name'], columns=book_new['book_name'])
print('Shape:', cosine_sim_df.shape)

# Melihat similarity matrix pada setiap buku
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""## Evaluasi

### Mendapatkan Rekomendasi
Membuat fungsi book_recommendations dengan beberapa parameter sebagai berikut:

* Nama_buku : Nama judul dari buku tersebut (index kemiripan dataframe).
* Similarity_data : Dataframe mengenai similarity yang telah kita didefinisikan sebelumnya
* Items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah ‘book_name’ dan ‘author’.
* k : Banyak rekomendasi yang ingin diberikan.
"""

def book_recommendations(nama_book, similarity_data=cosine_sim_df, items=book_new[['book_name', 'author']], k=5):


    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,nama_book].to_numpy().argpartition(
        range(-1, -k, -1))

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Drop nama_buku agar nama buku yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(nama_book, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)

book_new[book_new.book_name.eq('Wizard of Oz (Aladdin Classics)')]

# mendapatkan rekomendasi buku yang mirip dengan 'Wizard of Oz (Aladdin Classics)'
book_recommendations('Wizard of Oz (Aladdin Classics)')

"""# Collaborative Filtering

## Data Understanding
Supaya tidak tertukar dengan fitur ‘rating’ pada data, kita ubah nama variabel rating menjadi df
"""

# Membaca dataset
df = ratings
df

"""Dapat dilihat, data ratings memiliki 30000 baris dan 3 kolom

## Data Preparation

### Encode fitur User-ID dan ISBN
Melakukan persiapan data untuk menjadikan (encode) fitur ‘User-ID’ dan ‘ISBN’ ke dalam indeks integer
"""

# Mengubah User-ID menjadi list tanpa nilai yang sama
user_ids = df['User-ID'].unique().tolist()
print('list User-ID: ', user_ids)

# Melakukan encoding User-ID
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded User-ID : ', user_to_user_encoded)

# Melakukan proses encoding angka ke User-ID
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke User-ID: ', user_encoded_to_user)

# Mengubah ISBN menjadi list tanpa nilai yang sama
book_ids = df['ISBN'].unique().tolist()

# Melakukan proses encoding ISBN
book_to_book_encoded = {x: i for i, x in enumerate(book_ids)}

# Melakukan proses encoding angka ke ISBN
book_encoded_to_book = {i: x for i, x in enumerate(book_ids)}

"""### Memetakan User-ID dan ISBN
Petakan  User-ID dan ISBN ke dataframe yang berkaitan.
"""

# Mapping User-ID ke dataframe author
df['author'] = df['User-ID'].map(user_to_user_encoded)

# Mapping ISBN ke dataframe books
df['books'] = df['ISBN'].map(book_to_book_encoded)

"""### Cek data dan ubah nilai rating
Terakhir, cek beberapa hal dalam data seperti jumlah user, jumlah book, dan mengubah nilai rating menjadi float, cek nilai minimum dan maximum
"""

num_users = len(user_to_user_encoded)
print(num_users)

num_book = len(book_encoded_to_book)
print(num_book)

df['ratings'] = df['Book-Rating'].values.astype(np.float32)

min_rating = min(df['Book-Rating'])

max_rating = max(df['Book-Rating'])

print('Number of User: {}, Number of book: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_book, min_rating, max_rating
))

"""### Membagi data untuk latih dan validasi

membagi data latih dan validasi dengan komposisi 80:20
"""

# Mengacak dataset
df = df.sample(frac=1, random_state=42)
df

# Membuat variabel x untuk mencocokkan data author dan books menjadi satu value
x = df[['author', 'books']].values

# Membuat variabel y untuk membuat ratings dari hasil
y = df['ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

"""## Modelling

### Proses Latih

Membuat class RecommenderNet dengan keras Model class. Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation
"""

class RecommenderNet(tf.keras.Model):

  # Insialisasi fungsi
  def __init__(self, num_users, num_book, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_book = num_book
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.book_embedding = layers.Embedding( # layer embeddings book
        num_book,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.book_bias = layers.Embedding(num_book, 1) # layer embedding book bias

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    book_vector = self.book_embedding(inputs[:, 1]) # memanggil layer embedding 3
    book_bias = self.book_bias(inputs[:, 1]) # memanggil layer embedding 4

    dot_user_book = tf.tensordot(user_vector, book_vector, 2)

    x = dot_user_book + user_bias + book_bias

    return tf.nn.sigmoid(x) # activation sigmoid

model = RecommenderNet(num_users, num_book, 50) # inisialisasi model

# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# Memulai training

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 50,
    validation_data = (x_val, y_val)
)

"""## Evaluasi

Evaluasi dilakukan dengan menggunakan matriks Root Mean Squared Error (RMSE). Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Mean Square Error yaitu menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.
"""

# Visualisasi RMSE
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""Proses training model cukup smooth dan model konvergen pada epochs sekitar 50. Dari proses ini, kita memperoleh nilai error akhir sebesar sekitar 0.15 dan error pada data validasi sebesar 0.35 . Nilai tersebut cukup bagus untuk sistem rekomendasi.

### Mendapatkan Rekomendasi

Untuk mendapatkan rekomendasi buku, pertama kita ambil sampel user secara acak dan definisikan variabel book_not_read yang merupakan daftar buku yang belum pernah dibaca oleh pengguna
"""

book_df = book_new
df = pd.read_csv('/content/Ratings.csv')
# mengambil data sebanyak 30000
df = df.iloc[:30000,:]

user_id = df['User-ID'].sample(1).iloc[0]
book_read_by_user = df[df['User-ID'] == user_id]


book_not_read = book_df[~book_df['id'].isin(book_read_by_user.ISBN.values)]['id']
book_not_read = list(
    set(book_not_read)
    .intersection(set(book_to_book_encoded.keys()))
)

book_not_read = [[book_to_book_encoded.get(x)] for x in book_not_read]
user_encoder = user_to_user_encoded.get(user_id)
user_book_array = np.hstack(
    ([[user_encoder]] * len(book_not_read), book_not_read)
)

# untuk memperoleh rekomendasi buku, gunakan fungsi model.predict() dari library Keras
ratings = model.predict(user_book_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_book_ids = [
    book_encoded_to_book.get(book_not_read[x][0]) for x in top_ratings_indices
]

print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('book with high ratings from user')
print('----' * 8)

top_book_user = (
    book_read_by_user.sort_values(
        by = 'Book-Rating',
        ascending=False
    )
    .head(5)
    .ISBN.values
)

book_df_rows = book_df[book_df['id'].isin(top_book_user)]
for row in book_df_rows.itertuples():
    print(row.book_name, ':', row.author)

print('----' * 8)
print('Top 10 book recommendation')
print('----' * 8)

recommended_book = book_df[book_df['id'].isin(recommended_book_ids)]
for row in recommended_book.itertuples():
    print(row.book_name, ':', row.author)