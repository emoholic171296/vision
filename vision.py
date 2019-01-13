import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv("your location to image.csv file...").values
x_labels=data[0:21000,0]
x_features=data[0:21000,1:]
x_test=data[21000:,1:]
actual=data[21000:,0]
clf=DecisionTreeClassifier()
clf.fit(x_features,x_labels)
p=clf.predict(x_test)
count=0
for k in range(0,21000):
	if p[k]==actual[k]:
		count+=1
print("accuracy in percentage=",(count/21000)*100)



