import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x=["x","y","z"]
y=[5,7,2]
list1=["tran van tu ","Giang Le Hoang","Ha Thi Hue","Giang Thi Diep","tran van tu","Pham tuan Duong"]
list2=[30,20,22,20,22,22]
list3=["M","M","F","F","F","M"]
list4=[8,0,0,0,3,5]
data_in={"Họ và tên":list1, "Tuổi":list2, "Giới tính":list3,"Điểm toán":list4 }
s=pd.DataFrame(data_in)
#Trenf thêm điểm văn
s["Điểm văn"]=[5, 3,5 ,2,3,5]
diem_anh=[7,5,8,5,3,9]
#trèn thêm 1 cột điểm anh
s.insert(loc=4, column="Điểm anh ", value=diem_anh)
s.to_excel("Diem_tong_hop.xlsx", index=False)
# print(s)
df=pd.read_excel("Diem_tong_hop.xlsx")
print("df la:\n",df)
#Tạo biểu đồ dạng bar
# plt.bar(x,y)
#Tạo biểu đồ rải rác
# plt.scatter(x,y)
#Biểu đồ tần suất
# df.hist()
#Biểu đồ đường thẳng
# df.plot()
#Biểu đồ box
# df.boxplot()
# plt.show()
# plt.close()
x1=np.linspace(-10,10,100)
y1=np.sin(x1)
y2=np.cos(x1)
y3=x1**2
y4=x1**3
#vẽ biểu đồ hình sin
# plt.plot(x1,y1,label="Đồ thị hình sin ")
#Chú thích biểu đồ
# plt.title("Sóng siêu âm ")
# Nhãn biên độ
# plt.xlabel("Biên độ ")
# plt.ylabel("Thời gian ")
#Lưới
# plt.grid(True)
#Văn bản chú thích hình ảnh
# plt.figtext(0.01,-0.01,"Hinh 1: Sóng siêu âm ở tần số 1Hz")
# plt.legend(loc=1)
#Lưu ảnh về
# plt.savefig("Songsieuam.png")
# plt.plot(x1,y1,"g+", label="sin")
# plt.plot(x1,y2,"y-.", label="cos")
#Gán nhán bên trên loc=1
# plt.legend(loc=1)
# plt.axis([-5,20,-2,2])
# plt.subplot(2,2,1); plt.plot(x1,y1,"g *", label="sine"); plt.legend(loc=1)
# plt.subplot(2,2,2); plt.plot(x1,y2,"r *", label="cos"); plt.legend(loc=1)
# plt.subplot(2,2,3); plt.plot(x1,y3,"+", label="y=x**2"); plt.legend(loc=1)
# plt.subplot(2,2,4); plt.plot(x1,y4,"b--", label="y**3");plt.legend(loc=2)
#ảnh 1
plt.figure(1)
plt.subplot2grid((3,3), (0, 0), rowspan=2, colspan=2); plt.plot(x1,y1,"g+")
plt.subplot2grid((3,3), (0, 2),  colspan=2); plt.plot(x1,y2,"r-.")
#ảnh 2
plt.figure(2)
plt.subplot2grid((3,3), (2, 0),  colspan=2); plt.plot(x1,y3,"b *")
plt.subplot2grid((3,3), (1, 2),  rowspan=2, colspan=2);plt.plot(x1,y4,"y.")
#ảnh 3
fig1,sub1=plt.subplots()
sub1.pie(list4)
plt.show()