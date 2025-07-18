#Học máy có giám sát (Phân loại)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from openpyxl.styles.numbers import FORMAT_NUMBER_00
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix,classification_report, roc_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
#"Step1: cbi data"
X=np.array([[2,3,4,5,6,7,8,9,10]]).T
Y=np.array([0,0,0,0,0,1,1,1,1])
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,train_size=0.6, random_state=1)
#"Step2: Lựa chọn model "
moHinh=LogisticRegression()
#"Step3:Tranining model =>a, b"
moHinh.fit(X_train,Y_train)
#"Step4: Test data"
Y_train_p=moHinh.predict(X_train)
R2_train=r2_score(Y_train, Y_train_p)
print("R2 train:", R2_train)
Y_test_p=moHinh.predict(X_test)
R2_test=r2_score(Y_test, Y_test_p)
print("R2 test: ", R2_test)
# plt.plot(X_test,Y_test_p,"g*", label="True value")
# plt.plot(X_test,Y_test,"rs", label="Predict value")
# plt.legend(loc=0)
print("===================")
print("y test", Y_test)
print("Y test p:",Y_test_p)
print("Y_train   :", Y_train)
print("y_train_p :", Y_train_p)
# y1=moHinh.predict([[2]])
# print(y1)
# plt.show()
#Đánh giá mô hình
#1, Accuracy (Trả về độ chính xác phân loại)
#Score(X,Y)
accuracy_train=moHinh.score(X_train, Y_train)
print("R2_train=", accuracy_train)
accuracy_test=moHinh.score(X_test, Y_test)
print("R2_test=", accuracy_train)
#accuracy_score(Y,Yp)
accuracy_test2=accuracy_score(Y_test,Y_test_p)
print("accuracy_score=",accuracy_test2)
#Confusion matrix Bao gôm negative(-) postive(+)
CM=confusion_matrix(Y_train, Y_train_p)
TN=CM[0][0]
TP=CM[1][1]
FP=CM[0][1]
FN=CM[1][0]
print("Confusion_train=\n", CM)
#Phân tích chỉ số Precision và Recall
# precision=TP/(TP+FP) == Độ chính xác
# recal=TP/(TP+FN)     == Kết quả có bỏ sót trường hợp nào không
#Phân tích bài toán tìm F0
# Y: 100 ca âm tính, 15 ca F0
# Yp: 11 ca F0(6 ca f0 thật và 5 ca âm tính thật)  và 104 ca âm tính ( 9 ca F0 và 95 ca âm tính thật )
#Bản báo cáo report
report=classification_report(Y_train, Y_train_p)
print("Report:\n", report)
# Chỉ số 0<=F1-score=2*(pre.recall)/(pre+recall)<=1
# Hàm dự đoán xác xuất predic_proba
Y_train_pp=moHinh.predict_proba(X_train)[:,:]
print("Y_predict_proba ( xác xuất ) \n", Y_train_pp)

# Sensitivity (độ nhạy ) giông vs recall trả về keết quả có bỏ sót th nao ko = TPR
# FPR ( Báo động nhầm ), độ đặc hiệu + FPR = 1

#Đường cong ROC - Receiver Operating characteristics (Đặc điểm của bộ nhận )
#Chỉ số AUC để đánh giá là phần diện tích nằm ở phía dưới đường cong AUC càng lớn càng tốt
fpr, tpr, _=roc_curve(Y_test, Y_test_p)
plt.figure()
plt.plot(fpr, tpr,label="ROC curve")
plt.xlabel('False positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()
#============Tóm tắt các bước MKL có giám sát==================
# 1. Trainning & validation( train và xác thực )
# data set=> tiền xử lý => split( chia dữ liệu ) + train data set(70%) + test data set(30%)
# => đưa train data set vào model (ML) => Sử dụng hàm predict / predict_proba để kiểm tra => validaytion(xác thực )
# 2. Prediction(Dự đoán)
# New data => tiền xử lý => ML (predict)=> đưa ra kết quả dự đoán