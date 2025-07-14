import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import keras
import sklearn.linear_model as lm
from scipy.odr import polynomial
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
# a=np.array([0,2,3,5,6,7])
# b=np.array([5,11,14.,20,23,26])

# plt.plot(a,b)
# plt.show()
# mo_hinh=keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# mo_hinh.compile(optimizer="sgd", loss="mean_squared_error")
# mo_hinh.fit(a,b,epochs=100)
# b1 = mo_hinh.predict(np.array([[11]]))  # Đảm bảo input có shape (1, 1)
# print(b1)

#=================Chỉ số MSE (Mean Squared Error)===========================
#Đánh giá chất lượng của một dự đoán hoặc 1 ước lượng tương đương
#
# X=np.array([0,2,3,5,6,7])
# Y=np.array([5,11,14,20,23,26])
# Yp=np.array([6,13,15,18,20,28])
# saiSo=Y-Yp
# binhPhuongSaiSo=saiSo*saiSo
# tong=binhPhuongSaiSo.sum()
# MSE=tong/len(Y)
# print(MSE)

# #================Regression Hồi quy=======================
# so_gio_hoc=[2,3,4,5,6,7,8,9,10]
# diem=[30,35,40,44,48,60,66,70,75]
#
# data=pd.DataFrame({'So_gio_hoc': so_gio_hoc,'Diem':diem})
# data.to_excel('Du_lieu_hoc.xlsx', index=False)
# df=pd.read_excel('Du_lieu_hoc.xlsx')
#Tìm mối tương quan
# print(df.corr())
#========== Các bước thực hiện ML =============
# Y=aX + b
#
# B1: Cbi dữ liệu
df=pd.read_excel('Du_lieu_hoc.xlsx')
X=np.array([df.So_gio_hoc.values]).T
Y=df.Diem.values
# B2: Lựa chọn thuật toán
LR= lm.LinearRegression()
#Huấn luyện mô hình hồi quy đa thức
mo_hinh=make_pipeline(PolynomialFeatures(3),LR)
# B3: Trainning ->a , b
# LR.fit(X,Y)
mo_hinh.fit(X,Y)
# a=LR.coef_
# b=LR.intercept_
# print(a)
# print(b)
#Y=5,88*X + 16.999

# B4: Test data
# Yp=LR.predict(X)
Yp=mo_hinh.predict(X)
# # print(Y)
# print(np.int16(Yp))
# plt.plot(X,Y,"g*")
# plt.plot(X,Yp,"r--")
# plt.show()
# B5: Sử dụng predict
# print(LR.predict([[40]]))
#Chỉ số đánh giá lỗi MAE đo lường trung bình độ lệch tuyệt đối giữa dự đoán và thực tế.
MAE=mean_absolute_error(Y,Yp)
print("Chỉ số MAE:",MAE)
#Chỉ số MSE đo trung bình bình phương sai số, nên nhấn mạnh các sai số lớn hơn do chúng được bình phương.
MSE=mean_squared_error(Y,Yp)
print("Chỉ số MSE:",MSE)
#Chỉ số RMSE Được sử dụng phổ biến vì dễ hiểu hơn MSE, nhưng nhạy cảm với outliers như MSE.
RMSE=np.sqrt(MSE)
print("Chỉ số RMSE: ", RMSE)
#Chỉ số R-Square Cho biết tỷ lệ phương sai của biến mục tiêu được giải thích bởi mô hình.
R2=r2_score(Y,Yp)
print("Chỉ số R2:",R2)
#====Overfit và underfit=========
# 1.Overfit
# Mô hình học quá kỹ dữ liệu huấn luyện, bao gồm cả nhiễu và chi tiết không quan trọng. Điều này khiến nó hoạt động rất tốt trên tập huấn luyện nhưng kém trên tập kiểm tra.
# Học tốt + vận dụng kém
# 2. Underfit
# Mô hình quá đơn giản, không đủ khả năng học được các quy luật từ dữ liệu. Nó hoạt động kém trên cả tập huấn luyện và tập kiểm tra.
# Học dốt