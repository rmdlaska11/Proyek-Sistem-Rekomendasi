# Laporan Proyek Machine Learning - Rahmad Ramadhan Laska

## Domain Proyek

Buku menyimpan berbagai informasi yang mencakup iptek, seni budaya, ekonomi, politik, sosial, hingga pertahanan. Membaca buku tidak hanya menambah pengetahuan, tetapi juga membuka wawasan intelektual, memperkaya diri, dan mencegah kenakalan anak-anak. Sayangnya, Indonesia memiliki minat baca rendah, padahal sebagai negara besar, kita punya potensi untuk menjadi unggul. Membina minat baca sejak dini penting, karena anak-anak yang terbiasa membaca memiliki peluang lebih besar untuk mengembangkan pengetahuan mereka [1].


Sistem rekomendasi merupakan aplikasi yang memberikan saran kepada pengguna untuk membantu mereka membuat keputusan yang diinginkan. Untuk memberikan rekomendasi produk, sistem ini menggunakan filter data dengan mempertimbangkan faktor-faktor seperti perilaku pengguna, deskripsi produk, serta preferensi dan kebiasaan kelompok pengguna yang memiliki kesamaan dalam menilai suatu produk. Penelitian sebelumnya oleh Rizqi dan Arrie (2021) menunjukkan bahwa penggunaan sistem rekomendasi berbasis *deep learning* dapat meningkatkan kinerja dan kepuasan pengguna aplikasi.[2].
Untuk mengatasi permasalahan tersebut, dikembangkanlah sistem rekomendasi buku.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah dijelaskan diatas, terdapat beberapa masalah yaitu:
- Berdasarkan data mengenai pengguna, bagaimana membuat sistem rekomendasi buku yang dipersonalisasi dengan teknik *content-based filtering*?
- Dengan data rating yang dimiliki, bagaimana merekomendasikan buku lain yang mungkin disukai dan belum pernah dibaca oleh pengguna? 

### Goals

Untuk menjawab masalah yang ada, akan dibuat model prediksi dengan tujuan sebagai berikut:
- Menghasilkan sejumlah rekomendasi buku yang dipersonalisasi untuk pengguna dengan teknik *content-based filtering*.
- Menghasilkan sejumlah rekomendasi buku yang sesuai dengan preferensi pengguna dan belum pernah dibaca sebelumnya dengan teknik *collaborative filtering*.

### Solution statements
Solusi yang dapat dilakukan untuk sistem rekomendasi buku dengan menggunakan 2 algoritma *machine learning* yaitu:
- ***Content-based filtering*** : merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.
- ***Collaborative filtering*** : bergantung pada pendapat komunitas pengguna. Ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem berbasis konten.

Algoritma *content based filtering* digunakan untuk merekomendasikan buku berdasarkan aktivitas pengguna pada masa lalu, sedangkan algoritma *collaborative filtering* digunakan untuk merekomendasikan buku berdasarkan *rating* yang paling tinggi.

## Data Understanding
Data atau dataset yang digunakan pada proyek *machine learning* ini adalah data *Book Recommendation* dataset yang didapat dari situs Kaggle. 
Terdapat 3 file data dalam dataset. 
Link dataset dapat dilihat dari tautan berikut : [*Book Recommendation* Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).


### Variabel-variabel pada *Book Recommendation* dataset adalah sebagai berikut:  

- books : merupakan daftar buku tersebut.
- ratings : merupakan daftar penilaian yang diberikan pengguna terhadap buku.
- users : merupakan daftar pengguna.

Untuk memahami *Book Recommendation* dataset akan menggunakan beberapa teknik *Univariate Explanatory Data Analysis (EDA)* pada variabel-variabel berikut:
1.   Variabel books

Tabel 1. Info variabel books

| # | Column              | Non-Null Count  | Dtype  |
|---|---------------------|-----------------|--------|
| 0 | ISBN                | 271360 non-null | object |
| 1 | Book-Title          | 271360 non-null | object |
| 2 | Book-Author         | 271360 non-null | object |
| 3 | Year-Of-Publication | 271360 non-null | object |
| 4 | Publisher           | 271360 non-null | object |
| 5 | Image-URL-S         | 271360 non-null | object |
| 6 | Image-URL-M         | 271360 non-null | object |
| 7 | Image-URL-L         | 271360 non-null | object |

Pada Tabel 1, dapat dilihat bahwa:
*   Terdapat 271360 data dalam books
*   Terdapat 8 buah kolom bertipe objek yaitu ISBN, Book-Title, Book-Author,Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M dan Image-URL-L.


2.   Variabel users

Tabel 2. Info variabel users

| # | Column   | Non-Null Count  | Dtype   |
|---|----------|-----------------|---------|
| 0 | User-ID  | 278858 non-null | int64   |
| 1 | Location | 278858 non-null | object  |
| 2 | Age      | 278858 non-null | float64 |


Pada Tabel 2, dapat dilihat bahwa:

*   Terdapat 278858 data dalam users
*   Terdapat 1 buah kolom bertipe int64 yaitu User-ID.
*   Terdapat 1 buah kolom bertipe object yaitu Location.
*   Terdapat 1 buah kolom bertipe float64 yaitu Age.


3.   Variabel ratings

Variabel ratings memiliki jumlah data sebanyak 1149780 data. Karena data terlalu banyak, maka data yang akan digunakan hanya 30000 data saja. 

Tabel 3. Info variabel ratings

| # | Column         | Dtype   |
|---|----------------|---------|
| 0 | User-ID        | int64   |
| 1 | ISBN           | int64   |
| 2 | Book-Rating    | float64 |

Pada Tabel 3, dapat dilihat bahwa:

*   Terdapat 2 buah kolom bertipe int64 yaitu User-ID dan Book-rating.
*   Terdapat 1 buah kolom bertipe object yaitu ISBN.
  

## Data Preparation
Dalam proses persiapan data dibagi menjadi 2 berdasarkan algoritma yang digunakan yaitu pada *content-based filtering* dan *collaborative filtering* . Berikut tahapan-tahapan persiapan data:
***Content-based filtering***

- **Menangani *missing value*** : mengecek data apakah data tersebut ada yang bernilai NaN atau tidak, jika terdapat data yang kosong maka akan dihapus menggunakan fungsi dropna. Hal ini dilakukan karena missing value akan memengaruhi kinerja dan harus ada langkah khusus yang perlu diambil untuk mengatasinya.
- **Mengurutkan data** : mengurutkan data secara *ascending*. Hal ini dilakukan karena data yang terurut akan terlihat lebih rapih.
- **Menangani duplikat data** : Menghapus data yang duplikat dengan fungsi drop_duplicates(). Dalam hal ini, membuang data duplikat pada kolom ‘ISBN’. Hal ini dilakukan karena data duplikat memiliki informasi yang sama sehingga apabila dihapus tidak akan mempengaruhi kinerja.
- **Konversi data menjadi list** : Melakukan konversi data *series* menjadi *list*. Dalam hal ini, menggunakan fungsi tolist() dari *library numpy*. Hal ini dilakukan untuk menyederhanakan data menjadi bentuk *list*.
- **Membuat Dictionary** : Membuat *dictionary* untuk menentukan pasangan *key-value* pada data book_id, book_name, book_author dan book_publisher yang telah disiapkan sebelumnya. Hal ini dilakukan untuk persiapan data sebelum model dilatih.


***Collaborative filtering***

- ***Encode*** **fitur User-ID dan ISBN** : Melakukan persiapan data untuk menjadikan (*encode*) fitur ‘User-ID’ dan ‘ISBN’ ke dalam indeks integer. Hal ini diperlukan agar data siap digunakan untuk pemodelan.
- **Memetakan User-ID dan ISBN** : Petakan ‘User-ID’ dan ‘ISBN’ ke dataframe yang berkaitan. Hal ini diperlukan agar data yang sudah di *encode* dipetakan kemudian dimasukan kedalam dataframe yang berkaitan.
- **Cek data dan ubah nilai rating**: cek beberapa hal dalam data seperti jumlah *user*, jumlah *book*, dan mengubah nilai *rating* menjadi *float*, cek nilai *minimum* dan *maximum*. Hal ini dilakukan untuk mengecek data yang sudah siap digunakan untuk pemodelan.
- **Membagi data untuk latih dan validasi** : Membagi dataset menjadi data latih dan data validasi dengan perbandingan 80:20 yaitu 80 persen data akan menjadi data latih dan 20 persen data akan menjadi data validasi . Hal ini dilakukan supaya kita dapat melakukan validasi dengan benar tanpa bias dari model.

## Modeling

Pada tahap ini, model *machine learning* yang akan dipakai ada 2 algoritma. Berikut algoritma yang akan digunakan:

1.   *Content-based filtering* : algoritma *content-based filtering* dibuat dengan apa yang disukai pengguna pada masa lalu.
		- **kelebihan** : Model tidak memerlukan data tentang pengguna lain, karena rekomendasi bersifat khusus untuk pengguna ini. Hal ini mempermudah penskalaan ke sejumlah besar pengguna.
		- **kekurangan** : Model hanya dapat membuat rekomendasi berdasarkan minat pengguna yang ada. Dengan kata lain, model memiliki kemampuan terbatas untuk memperluas minat pengguna yang ada.
2.   *Collaborative filtering* : algoritma *collaborative filtering* dibuat dengan memanfaatkan tingkat *rating* dari buku tersebut. 
		- **kelebihan** : Tidak memerlukan pengetahuan domain dan model dapat membantu pengguna menemukan minat baru..
		- **kekurangan** :Tidak dapat menangani item baru dan sulit menyertakan fitur samping untuk kueri/item.

Hasil dari masing-masing model:

1. ***Content-based filtering***

Berikut adalah buku yang disukai pengguna di masa lalu:

Tabel 4. buku yang disukai pengguna di masa lalu.

| id          |       book_name                   |   author     | publisher |
|------------:|----------------------------------:|-------------:|----------:|
|  0689831420 | 'Wizard of Oz (Aladdin Classics)' | L. Frank Baum| Aladdin   |

Pada tabel 4, dapat dilihat bahwa pengguna menyukai buku yang berjudul 'Wizard of Oz (Aladdin Classics)') yang ditulis oleh L.Frank Baum. Maka hasil 5 rekomendasi terbaik berdasarkan algoritma *content-based filtering* adalah sebagai berikut :

Tabel 5. Hasil rekomendasi algoritma *content-based filtering*

|   |                       book_name                   | author        |
|--:|--------------------------------------------------:|--------------:|
| 0 | Dorothy and the Wizard of Oz (Complete and Una..) | L. Frank Baum |
| 1 | Der Zauberer Von Oos                              | Frank L. Baum |
| 2 | The Wizard of Oz                                  | L. Frank Baum |
| 3 | Aerie Tik Tok of Oz: Defiant-Cn16dp               | Baum          |
| 4 | The Diary of a Young Girl: The Definitive Edition | Anne Frank    | 

Dapat dilihat pada tabel 5, ada 5 buku yang direkomendasikan penulisnya yang sama yaitu  L. Frank Baum. Hal ini didasarkan pada kesukaan pembaca atau pengguna pada masa lalu.

2. ***Collaborative filtering***

Berikut merupakan buku berdasarkan *rating* yang ada :

![Prediksi CF](https://github.com/rmdlaska11/Proyek-Sistem-Rekomendasi/assets/121273531/5ec53a15-b0f7-478f-96ac-6a58bf744b90)

Gambar 1. Hasil rekomendasi algoritma *collaborative filtering*

Pada Gambar 1, Buku yang memiliki *rating* tinggi dari pengguna paling banyak ditulis oleh James Finn Garner

## Evaluation
Hasil evaluasi dari masing-masing model:

1. ***Content-based filtering***

Buku yang direkomendasikan di masa lalu yaitu 'Wizard of Oz (Aladdin Classics)' dengan dengan penulis  L. Frank Baum. Top 5 item hasil rekomendasi buku semua memilki dengan penulis yang memiliki nama yang hampir sama yaitu  L. Frank Baum. Dengan begitu hasil rekomendasi dapat dievaluasi menggunakan rumus presisi berikut :

*recommender system precision*: P = $\frac{n of our recommendations that are relevant}{n of items we recommended}$

Dengan begitu hasil presisi yang didapat adalah 100 persen .

2.  ***Collaborative filtering***

Evaluasi metrik yang digunakan untuk mengukur kinerja model adalah metrik RMSE (Root Mean Squared Error). RMSE adalah metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil. Metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besa


Rumus dari matriks RMSE adalah sebagai berikut:

MSE = $\sqrt{\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2}$

Dimana :

y = Nilai Aktual permintaan

$\hat{y}$ = Nilai hasil peramalan

n = banyaknya data

Berikut visualisasi RMSE menggunakan *collaborative filtering*:

![RMSE](https://github.com/rmdlaska11/Proyek-Sistem-Rekomendasi/assets/121273531/b21f9d0d-af87-4550-adf0-385a89fc2af2)


Gambar 2. Visualisasi RMSE 

Bisa dilihat pada Gambar 2. proses training model cukup smooth dan model konvergen pada epochs sekitar 50. Dari proses latih, memperoleh nilai error akhir sebesar sekitar 0.15 dan error pada data validasi sebesar 0.35 . Nilai tersebut cukup bagus untuk sistem rekomendasi.

**Kesimpulan** :  Berhasil menghasilkan 5 rekomendasi buku terbaik menggunakan teknik *content-based filtering* dengan presisi 100 persen dan berhasil menghasilkan 10 rekomendasi buku terbaik menggunakan teknik *collaborative filtering* dengan RMSE sebesar 0.15 .

**Referensi** :

[1]	M. Irfan, A. D. Cahyani, and F. H. R, “SISTEM REKOMENDASI: BUKU ONLINE DENGAN METODE COLLABORATIVE FILTERING”, TECHNOSCIENTIA, vol. 7, no. 1, pp. 076–84, Aug. 2014.

[2]	M. R. A. Zayyad, A. Kurniawardhani, S. Si, dan M. Kom, “Penerapan Metode Deep Learning pada Sistem Rekomendasi Film,” hlm. 5.


