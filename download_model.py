import urllib.request
import bz2
import os

url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
extracted_file = "shape_predictor_68_face_landmarks.dat"

print("Downloading shape_predictor_68_face_landmarks.dat...")
urllib.request.urlretrieve(url, compressed_file)

print("Extracting...")
with bz2.BZ2File(compressed_file, 'rb') as source, open(extracted_file, 'wb') as dest:
    dest.write(source.read())

# 刪除壓縮檔以節省空間 (可選)
os.remove(compressed_file)

print(f"Done! Model saved to: {extracted_file}")