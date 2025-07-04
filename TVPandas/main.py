import numpy as np
import pandas as pd
# m1 = np.int16(([[2,4,6,8],[3,2,0,1]]))
# m2 = np.int16(([[20,40,60,80],[30,20,10,60]]))
# # m1=np.array([1,2,3,4])
# # m2=np.array([5,6,7,8])
# print("m1 là: \n ",m1 )
# print("m2 là: \n ",m2 )
# np.savetxt("my_array.csv",m1)
# m3=np.loadtxt("my_array.csv")
# m4=np.loadtxt("my_array.txt")
# print("m3 la:\n", m3)
# print("m4 la:\n", m4)

# ng= np.random.default_rng(0)
# m= ng.integers(3,10,(3,4))
# print(m)
# m1=np.flip(m,axis=1)
# print("m1 la :\n",m1)
list1=["tran van tu ","Giang Le Hoang","Ha Thi Hue","Giang Thi Diep","tran van tu"]
list2=[20,40,60,80,20]
list3=[15,20,31,42,15]
# m=np.array(list1)
# # print("M ang m la: ",m)
data_in={"Name":list1,"Gender":list2}
s=pd.DataFrame(data_in, columns=["Gender", "Name"])
# print(s)
# # print(s.iloc[0])
# # print(s.iloc[0:2])
# # del s["Name"]
s["score"]=list3
s["Gender"]=[1,2,3,4,5]
# # s.to_csv("mydf.txt",sep='\n')
# print(m)
print(s)
# s=pd.read_csv("mydf.csv", sep='\t')
# m=s[(s["Gender"]<=3) & (s["Name"]=="Giang Le Hoang")]
s.columns=["STT", "Ho ten", "Tuoi"]
s["New Age"]=s["Tuoi"]+10
s["Nam"]=s["New Age"]-s["Tuoi"]
m=s["New Age"].max()
print(s)
print(m)
