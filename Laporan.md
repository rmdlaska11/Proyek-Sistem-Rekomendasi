# Laporan Proyek Machine Learning - Rahmad Ramadhan Laska

## Domain Proyek

Buku menyimpan berbagai informasi yang mencakup iptek, seni budaya, ekonomi, politik, sosial, hingga pertahanan. Membaca buku tidak hanya menambah pengetahuan, tetapi juga membuka wawasan intelektual, memperkaya diri, dan mencegah kenakalan anak-anak. Sayangnya, Indonesia memiliki minat baca rendah, padahal sebagai negara besar, Indonesia punya potensi untuk menjadi unggul. Membina minat baca sejak dini penting, karena anak-anak yang terbiasa membaca memiliki peluang lebih besar untuk mengembangkan pengetahuan mereka [1].


Sistem rekomendasi merupakan aplikasi yang memberikan saran kepada pengguna untuk membantu mereka membuat keputusan yang diinginkan. Untuk memberikan rekomendasi produk, sistem ini menggunakan filter data dengan mempertimbangkan faktor-faktor seperti perilaku pengguna, deskripsi produk, serta preferensi dan kebiasaan kelompok pengguna yang memiliki kesamaan dalam menilai suatu produk. Penelitian sebelumnya oleh Rizqi dan Arrie (2021) menunjukkan bahwa penggunaan sistem rekomendasi berbasis *deep learning* dapat meningkatkan kinerja dan kepuasan pengguna aplikasi.[2].
Untuk mengatasi permasalahan tersebut, dikembangkanlah sistem rekomendasi buku.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah dijelaskan diatas, terdapat beberapa masalah yaitu:
- Bagaimana cara membuat sistem rekomendasi buku yang dapat dipersonalisasi berdasarkan data pengguna dengan menerapkan teknik content-based filtering?
- Bagaimana cara menggunakan data rating yang sudah ada untuk menghasilkan rekomendasi buku baru yang mungkin disukai oleh pengguna, sambil memastikan bahwa buku-buku tersebut belum pernah dibaca sebelumnya oleh pengguna?

### Goals

Untuk menjawab masalah yang ada, akan dibuat model prediksi dengan tujuan sebagai berikut:
- Mengembangkan model content-based filtering yang mampu menghasilkan rekomendasi buku yang sesuai dengan preferensi unik setiap pengguna. Keberhasilan model akan diukur melalui akurasi dan relevansi rekomendasi.
- Membangun model collaborative filtering yang dapat memanfaatkan data rating untuk mengidentifikasi dan merekomendasikan buku-buku yang mungkin disukai oleh pengguna, dengan memastikan bahwa buku-buku tersebut belum pernah diakses oleh pengguna sebelumnya. Keberhasilan model akan diukur melalui keakuratan dan keberagaman rekomendasi.

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
  
Deskripsi dari variabel ratings dapat dilihat pada tabel 4 berikut:

Tabel 4. Deskripsi variabel ratings

|       |      User-ID |   Book-Rating |     
|------:|-------------:|---------------|
| count | 30000.000000 |  30000.000000 |
|  mean | 91215.561800 |      3.062533 |
|   std | 127794.30475 |      3.901264 |
|   min |     2.000000 |      0.000000 |
|   25% |  2977.000000 |      0.000000 |
|   50% |  5970.000000 |      0.000000 |
|   75% | 277478.00000 |      7.000000 |
|   max | 278854.00000 |     10.000000 |

Pada tabel 4, dapat dilihat dari nilai max dan min bahwa nilai rating terbesar yaitu 10 dan nilai rating terkecil yaitu 0

## Data Preparation
Dalam proses persiapan data dibagi menjadi 2 berdasarkan algoritma yang digunakan yaitu pada *content-based filtering* dan *collaborative filtering* . Berikut tahapan-tahapan persiapan data:
***Content-based filtering***

- **Menangani *missing value*** : Mengecek apakah ada nilai NaN dalam data; jika ada, data tersebut akan dihapus menggunakan fungsi dropna. Langkah ini diperlukan karena keberadaan nilai yang hilang dapat memengaruhi kinerja dan memerlukan tindakan khusus untuk penanganannya, terutama jika nama bukunya tidak dapat diidentifikasi.
- **Mengurutkan data** : mengurutkan data secara *ascending*. Hal ini dilakukan karena data yang terurut akan terlihat lebih rapih.
- **Menangani duplikat data** : Menghapus data yang duplikat dengan fungsi drop_duplicates(). Dalam hal ini, membuang data duplikat pada kolom ‘ISBN’. Hal ini dilakukan karena data duplikat memiliki informasi yang sama sehingga apabila dihapus tidak akan mempengaruhi kinerja.
- **Konversi data menjadi list** : melakukan konversi data *series* menjadi *list* dengan menggunakan fungsi tolist() dari library numpy. Tujuan dari langkah ini adalah untuk menyederhanakan representasi data menjadi bentuk *list*, mempermudah proses pengolahan data lebih lanjut.
- **Membuat Dictionary** : membuat *dictionary* yang akan menentukan pasangan *key-value* pada data book_id, book_name, book_author, dan book_publisher yang telah disiapkan sebelumnya. Pada tahap ini, data diatur dalam bentuk pasangan atribut buku dan nilainya, menciptakan representasi yang lebih terstruktur untuk persiapan data sebelum dilakukan pelatihan model.


***Collaborative filtering***

- ***Encode*** **fitur User-ID dan ISBN** : melakukan persiapan data dengan melakukan encoding pada fitur ‘User-ID’ dan ‘ISBN’. Proses ini bertujuan untuk mengubah representasi data ke dalam indeks integer, memastikan bahwa data siap untuk digunakan dalam pemodelan. Encoding diperlukan agar model dapat memproses dan memahami fitur-fitur ini dengan lebih efisien.
- **Memetakan User-ID dan ISBN** : memetakan ‘User-ID’ dan ‘ISBN’ ke dalam dataframe yang relevan. Ini penting agar data yang telah diencode dapat ditempatkan kembali dalam konteks aslinya. Pemetaan ini membantu menjaga keterkaitan data dan memudahkan penggunaan data dalam model.
- **Cek data dan ubah nilai rating**: Proses berikutnya mencakup pemeriksaan beberapa aspek dalam data, seperti jumlah pengguna (user), jumlah buku (book), dan pengubahan nilai rating menjadi tipe data float. Cek nilai minimum dan maksimum dilakukan untuk memastikan integritas data sebelum proses pemodelan dimulai.
- **Membagi data untuk latih dan validasi** :  membagi dataset menjadi data latih dan data validasi dengan perbandingan 80:20. Sebanyak 80 persen data akan digunakan untuk melatih model, sementara 20 persen sisanya akan digunakan untuk validasi. Pembagian ini diperlukan untuk melakukan validasi tanpa bias dari model, sehingga model dapat dievaluasi dengan benar.
## Modeling

Pada tahap ini, model *machine learning* yang akan dipakai ada 2 algoritma. Berikut algoritma yang akan digunakan:

1. Content-based Filtering:
- Kelebihan: Model content-based filtering memberikan rekomendasi yang sangat personal, disesuaikan dengan minat pengguna berdasarkan preferensi masa lalu. Kelebihan ini sangat menguntungkan dalam penskalaan ke sejumlah besar pengguna, karena model tidak memerlukan data pengguna lain. Hal ini dapat meningkatkan efisiensi dan memudahkan implementasi pada platform dengan jumlah pengguna yang besar.
- Kekurangan: Meskipun model ini efektif dalam memberikan rekomendasi berdasarkan minat pengguna yang ada, kekurangannya terletak pada keterbatasan kemampuan untuk memperluas minat pengguna. Model ini cenderung memberikan rekomendasi yang terpaku pada minat yang sudah diketahui, kurang mampu menggali minat baru atau menghadirkan variasi dalam rekomendasi. Hal ini dapat membatasi pengguna dalam menjelajahi konten yang berbeda atau baru.

Pengaruh pada Keputusan Bisnis atau Pengalaman Pengguna:
- Keputusan Bisnis: Model content-based filtering dapat menjadi pilihan yang baik untuk bisnis dengan fokus pada personalisasi tinggi, terutama jika platform memiliki informasi detil mengenai preferensi pengguna. Namun, model ini mungkin kurang efektif untuk bisnis yang ingin mendorong eksplorasi produk atau menawarkan variasi kepada pengguna.
- Pengalaman Pengguna: Meskipun memberikan rekomendasi yang sesuai dengan minat pengguna, model ini dapat membuat pengalaman pengguna menjadi kurang dinamis. Pengguna mungkin merasa terjebak dalam lingkaran minat yang sama tanpa banyak variasi atau penemuan baru.

2. Collaborative Filtering:
- Kelebihan: Collaborative filtering tidak memerlukan pengetahuan domain dan dapat membantu pengguna menemukan minat baru berdasarkan preferensi pengguna serupa. Kelebihan ini memungkinkan model lebih adaptif terhadap perubahan minat pengguna seiring waktu, menciptakan pengalaman yang dinamis dan responsif.
- Kekurangan: Model ini memiliki kendala dalam menangani item baru, karena membutuhkan data tingkat rating yang cukup untuk item tertentu sebelum dapat memberikan rekomendasi yang akurat. Selain itu, kesulitan dalam menyertakan fitur samping (contohnya, genre buku) dapat membuat model kurang presisi dalam memahami preferensi pengguna.

Pengaruh pada Keputusan Bisnis atau Pengalaman Pengguna:
- Keputusan Bisnis: Collaborative filtering dapat menjadi pilihan yang baik untuk bisnis yang ingin mendorong interaksi sosial dan pertukaran rekomendasi antar pengguna. Namun, perlu diingat bahwa kendala terhadap item baru dapat membatasi efektivitas model, terutama dalam lingkungan dengan banyak produk baru.
- Pengalaman Pengguna: Model ini dapat memberikan pengalaman yang lebih dinamis dan interaktif bagi pengguna dengan menawarkan rekomendasi berdasarkan tingkat kesamaan dengan pengguna lain. Namun, keterbatasan dalam menangani item baru dapat membuat pengguna kehilangan potensi menemukan konten terbaru atau unik.

Hasil dari masing-masing model:

1. ***Content-based filtering***

Berikut adalah buku yang disukai pengguna di masa lalu:

Tabel 5. buku yang disukai pengguna di masa lalu.

| id          |       book_name                   |   author     | publisher |
|------------:|----------------------------------:|-------------:|----------:|
|  0689831420 | 'Wizard of Oz (Aladdin Classics)' | L. Frank Baum| Aladdin   |

Pada tabel 5, dapat dilihat bahwa pengguna menyukai buku yang berjudul 'Wizard of Oz (Aladdin Classics)') yang ditulis oleh L.Frank Baum. Maka hasil 5 rekomendasi terbaik berdasarkan algoritma *content-based filtering* adalah sebagai berikut :

Tabel 6. Hasil rekomendasi algoritma *content-based filtering*

|   |                       book_name                   | author        |
|--:|--------------------------------------------------:|--------------:|
| 0 | Dorothy and the Wizard of Oz (Complete and Una..) | L. Frank Baum |
| 1 | Der Zauberer Von Oos                              | Frank L. Baum |
| 2 | The Wizard of Oz                                  | L. Frank Baum |
| 3 | Aerie Tik Tok of Oz: Defiant-Cn16dp               | Baum          |
| 4 | The Diary of a Young Girl: The Definitive Edition | Anne Frank    | 

Dapat dilihat pada tabel 6, ada 5 buku yang direkomendasikan penulisnya yang hampir sama yaitu  L. Frank Baum. Hal ini didasarkan pada kesukaan pembaca atau pengguna pada masa lalu.

2. ***Collaborative filtering***

Berikut merupakan buku berdasarkan *rating* yang ada :

Rekomendasi Buku untuk Pengguna: 277427

Tabel 7. Buku dengan Rating Tinggi dari Pengguna:

| No |	Judul                                          | Buku	Penulis      |
|---:|------------------------------------------------:|------------------:|
| 1	 | Politically Correct Bedtime Stories	           | James Finn Garner |
| 2	 | Its Obvious You Wont Survive By Your Wit	       | Scott Adams       |
| 3	 | Unnatural Selections                            |	Gary Larson      |
| 4	 | Dilbert Fugitive From The Cubicle Police	       | Scott Adams       |
| 5	 | Dilbert: Seven Years Of Highly Defective People | Scott Adams       |

Tabel 8. Top 10 Rekomendasi Buku
| No |	Judul                                          | Buku	Penulis     |
|---:|------------------------------------------------:|-----------------:|
| 1	 | The Giver (21st Century Reference)		           | LOIS LOWRY       |
| 2	 | Angel Eyes                                      | Eric Lustbader   |
| 3	 | Reaper Man	                                     |	Terry Pratchett |
| 4	 | Queen of the Darkness (Black Jewels Trilogy)    | Anne Bishop      |
| 5	 | Blackberry Wine	                               | Joanne Harris    |
| 6	 | The Perks of Being a Wallflower		             | Stephen Chbosky  |
| 7	 | The Golden Compass (His Dark Materials, Book 1) | PHILIP PULLMAN   |
| 8	 | Kushiel's Dart		                               | Jacqueline Carey |
| 9	 | Yukon Ho!	                                     | Bill Watterson   |
| 10 | Comme un roman	                                 | Daniel Pennac    |

## Evaluation
Hasil evaluasi dari masing-masing model:

1. ***Content-based filtering***

Buku yang direkomendasikan di masa lalu yaitu 'Wizard of Oz (Aladdin Classics)' dengan dengan penulis  L. Frank Baum. Dari hasil evaluasi presisi, didapatkan nilai 75%. Presisi dihitung dengan membandingkan jumlah rekomendasi yang relevan dengan total rekomendasi yang diberikan. Dalam konteks ini, nilai 75% mengindikasikan bahwa dari 5 rekomendasi yang diberikan, 3 di antaranya relevan dengan preferensi pengguna.

2.  ***Collaborative filtering***

Evaluasi metrik yang digunakan untuk mengukur kinerja model adalah metrik RMSE (Root Mean Squared Error). RMSE adalah metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil. Metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besa


Rumus dari matriks RMSE adalah sebagai berikut:

MSE = $\sqrt{\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2}$

Dimana :

y = Nilai Aktual permintaan

$\hat{y}$ = Nilai hasil peramalan

n = banyaknya data

Berikut visualisasi RMSE menggunakan *collaborative filtering*:

![RMSE](https://github.com/rmdlaska11/Proyek-Sistem-Rekomendasi/assets/121273531/50469802-9bfc-4f12-b7bd-64f7913bb49d)



Gambar 2. Visualisasi RMSE 

Bisa dilihat pada Gambar 2. proses training model cukup smooth dan model konvergen pada epochs sekitar 50. Dari proses latih, memperoleh nilai error akhir sebesar sekitar 0.15 dan error pada data validasi sebesar 0.35 . Nilai tersebut cukup bagus untuk sistem rekomendasi.

**Kesimpulan** :  Cukup berhasil menghasilkan 5 rekomendasi buku terbaik menggunakan teknik *content-based filtering* dengan presisi 75 persen dan berhasil menghasilkan 10 rekomendasi buku terbaik menggunakan teknik *collaborative filtering* dengan RMSE sebesar 0.15 .

**Referensi** :

[1]	M. Irfan, A. D. Cahyani, and F. H. R, “SISTEM REKOMENDASI: BUKU ONLINE DENGAN METODE COLLABORATIVE FILTERING”, TECHNOSCIENTIA, vol. 7, no. 1, pp. 076–84, Aug. 2014.

[2]	M. R. A. Zayyad, A. Kurniawardhani, S. Si, dan M. Kom, “Penerapan Metode Deep Learning pada Sistem Rekomendasi Film,” hlm. 5.


