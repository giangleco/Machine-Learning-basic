#Thư viện Seaborn để vẽ biểu đồ thống kê trong python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from keras.src.losses import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
#Hàm load_dataset đê đọc dữ liệu

# tien_tips=sns.load_dataset("tips")
# tien_tips.to_csv("tien_tip.csv", index=True)
# print(tien_tips)
tien_tips=pd.read_csv('tien_tip.csv')
plt.figure(figsize=(8,4))
#Hàm histplot= Gom dữ liệu thành nhóm -> Đếm số lượng -> vẽ cột -> Giúp nhìn nhanh xem dữ liệu thường rơi vào khoảng nào
# (Ưu tiên số lượng nhỏ ) thống kê chính xác

# s=pd.Series([12, 15, 17,20, 22, 25, 27, 30,32, 50])
# s=pd.DataFrame({'Hoa don':[12, 15, 17,20, 22, 25, 27, 30,32, 50],'size':[1,2,3,4,5,6,7,8,9,10]})
plt.subplot(1,2,1)
sns.histplot(data=tien_tips,x='tip', bins=5, color="Skyblue", edgecolor="black", kde=True)


plt.title('Histogram + KDE: Phân phối tiền tip')
plt.xlabel("Tip (%$)")
plt.ylabel("Tần suất")

#Hàm kdeplot() - biểu đồ mật độ xác suất (ưu tiên số lượng lớn ) mượt mà hơn
plt.subplot(1,2,2)
sns.kdeplot(data=tien_tips,x="tip", fill=False, color="Skyblue", alpha=0.6, linewidth=2)

plt.title('KDE Only: Phân phối tiền tip')
plt.xlabel("Tip (%$)")
plt.ylabel("Tần suất")

#Hàm displot()- Biểu đồ phân phối
g=sns.displot(
    data=tien_tips,
    x="tip",
    kind="hist",
    col="sex",
    row="smoker",
    kde=True,
    bins=20,
    fill="True",
    height=3,       #chiều cao mỗi sublot: 2.5 inch
    aspect=1        #Mỗi subplot rộng 2.5*1 =2.5 inch =>figure ~ 2.5x2.5 inch
)

#Thêm tiêu đề toàn cục
g.fig.suptitle("Phân phối tiền tip (Histogram + KDE ", y=1.05)
# plt.show()

#Mã hóa labelEncoder
le = LabelEncoder()
tien_tips["gender_encoded"] = le.fit_transform(tien_tips['sex'])
#Mã hóa Onehotencoding

