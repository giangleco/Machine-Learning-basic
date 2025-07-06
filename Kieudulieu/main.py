#Kiểu dữ liệu văn bản

#tách câu văn bản
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
cau="Giang Le Hoang lop cntt02"
danhsach1=word_tokenize(cau)
print("Danh sach chua cac tu:", danhsach1)

#Thư viện dịch thuật
from transformers import pipeline
translator=pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en")
ketqua=translator("Xin chào, tôi tên là Hoang ")
print(ketqua)

#Kiểu dữ liệu thời gian
from statsmodels.tsa.arima.model import ARIMA

#Dữ liệu giả lập:
temperature_data=[
    25, 26, 27, 29, 30 ,
    32, 31, 34, 30, 28
]
# Tạo mô hình ARIMA
moHinh=ARIMA(temperature_data, order=(1,1,1))   #ARIMA(p=1, d=1, q=1)
moHinhFIt=moHinh.fit()

#Dự báo 3 giá trị tiếp theo
forecast=moHinhFIt.forecast(steps=3)

print("Dự báo 3 ngày tới:", forecast)