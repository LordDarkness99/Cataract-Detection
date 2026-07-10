Dataset:"https://www.kaggle.com/datasets/nandanp6/cataract-image-dataset"

untuk bisa menjalankan web deteksi katarak ini dapatkan model "vgg_cataract.onnx" caranya yaitu:
1. Jalankan file ipynb dan dapatkan modelnya dalam format .pth
2. Convert ke .onnx "sebenarnya ini opsional bisa langsung jalankan menggunakan .pth" tapi usahakan full version dan sedikt rubah format app.py
3. buat .ENV untuk menyimpan library yang dibutuhkan "jalankan requirements.txt"
4. Jalankan web dengan perintah "python app.py"
