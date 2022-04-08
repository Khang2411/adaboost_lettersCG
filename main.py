import numpy as np
import pandas as pd
import self as self
#train_test_splitchia các mảng hoặc ma trận thành các tập con ngẫu nhiên và kiểm tra.
# Điều đó có nghĩa là mỗi khi bạn chạy nó mà không chỉ định random_state, bạn sẽ nhận được một kết quả khác, đây là hành vi dự kiến.
data = pd.read_csv("letters_CG.csv")
from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(data, test_size=0.3311258278145696, random_state=1)
# lấy dữ liệu để training là 33,11...% = 500 = đề còn lại là test

X_train = training_set.iloc[:, 1:16].values
Y_train = training_set.iloc[:, 0].values
#print(Y_train)

X_test = test_set.iloc[:, 1:16].values # lấy coulmn từ 1->16
Y_test = test_set.iloc[:, 0].values # lấy label Class
from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(n_estimators=100, base_estimator=None, learning_rate=1, random_state = 1)
# base_estimator là thuật toán học tập được sử dụng để đào tạo các mô hình yếu.
# Điều này hầu như luôn luôn không cần phải thay đổi bởi vì cho đến nay người học phổ biến nhất sử dụng AdaBoost
# là một cây quyết định - đối số mặc định của tham số này.
# n_estimator là số lượng mô hình để đào tạo lặp đi lặp lại.
# learning-rate các giá trị nhỏ hơn 50% learning_rate thì học yếu -> thúc đầy để học tiếp bằng cách tăng max_dept
adaboost.fit(X_train, Y_train)
Y_pred = adaboost.predict(X_test)
test_set["Predictions"] = Y_pred
print(test_set.drop("Unnamed: 17", axis=1))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
accuracy = float(cm.diagonal().sum()) / len(Y_test)
print("\nĐộ chính xác của AdaBoost cho tập dữ liệu đã cho :" '', accuracy)
