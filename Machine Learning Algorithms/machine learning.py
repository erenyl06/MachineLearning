
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

heart = pd.read_csv('heart.csv')

heart[heart.duplicated()]
heart.drop_duplicates(keep='first',inplace=True)

print('Number of rows are',heart.shape[0], 'and number of columns are ',heart.shape[1])

heart.describe()
heart.corr()

fig = make_subplots(rows=3, cols=4,specs=[[{'type':'xy'},{'type':'xy'},{'type':'xy'},{'type':'xy'}],[{'type':'xy'},{'type':'xy'},{'type':'xy'},{'type':'xy'}],[{'type':'xy'},{'type':'xy'},{'type':'xy'},{'type':'xy'}]])

fig.add_trace(go.Histogram(x = heart['age'], name='Age'),row=1, col=1)
fig.add_trace(go.Histogram(x = heart['sex'],name='Sex'),row=1, col=2)
fig.add_trace(go.Histogram(x = heart['cp'],name='Chest Pain'),row=1, col=3)
fig.add_trace(go.Histogram(x = heart['trtbps'], name='Resting Blood Pressure'),row=1, col=4)
fig.add_trace(go.Histogram(x = heart['chol'], name='Cholestoral'),row=2, col=1)
fig.add_trace(go.Histogram(x = heart['fbs'], name='Fasting Blood Sugar'),row=2, col=2)
fig.add_trace(go.Histogram(x = heart['restecg'], name='Resting electrocardiographical results'),row=2, col=3)
fig.add_trace(go.Histogram(x = heart['thalachh'], name='Maximum heart rate'),row=2, col=4)
fig.add_trace(go.Histogram(x = heart['oldpeak'], name='Previous peak'),row=3, col=1)
fig.add_trace(go.Histogram(x = heart['caa'], name='Number of major vessels'),row=3, col=2)
fig.add_trace(go.Histogram(x = heart['thall'], name='Thalium stress test'),row=3, col=3)
fig.add_trace(go.Histogram(x = heart['exng'], name='Exercise induced angina'),row=3, col=4)


fig.update_xaxes(title_text='Age',title_font = {"size": 12, "color":"#338FC2"}, row=1, col=1)
fig.update_xaxes(title_text='Sex', title_font = {"size": 12, "color":"#338FC2"}, row=1, col=2)
fig.update_xaxes(title_text='Chest Pain', title_font = {"size": 12, "color":"#338FC2"}, row=1, col=3)
fig.update_xaxes(title_text='Resting Blood Pressure', title_font = {"size": 12, "color":"#338FC2"}, row=1, col=4)
fig.update_xaxes(title_text='Cholestoral', title_font = {"size": 12, "color":"#338FC2"}, row=2, col=1)
fig.update_xaxes(title_text='Fasting Blood Sugar', title_font = {"size": 12, "color":"#338FC2"}, row=2, col=2)
fig.update_xaxes(title_text='Resting electrocardiographical results', title_font = {"size": 12, "color":"#338FC2"}, row=2, col=3)
fig.update_xaxes(title_text='Maximum heart rate', title_font = {"size": 12, "color":"#338FC2"}, row=2, col=4)
fig.update_xaxes(title_text='Previous peak', title_font = {"size": 12, "color":"#338FC2"}, row=3, col=1)
fig.update_xaxes(title_text='Number of major vessels', title_font = {"size": 12, "color":"#338FC2"}, row=3, col=2)
fig.update_xaxes(title_text='Thalium stress test', title_font = {"size": 12, "color":"#338FC2"}, row=3, col=3)
fig.update_xaxes(title_text='Exercise induced angina', title_font = {"size": 12, "color":"#338FC2"}, row=3, col=4)


colors=["#eea05c"]
fig.update_layout(title="Histograms", title_font_color="#338FC2", colorway=colors)



x = heart.iloc[:, 1:-1].values
y = heart.iloc[:, -1].values
x,y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train,x_test


model1 = KNeighborsClassifier(n_neighbors=15)
accuracy1 = cross_val_score(model1, x, y, scoring='accuracy', cv = 10)
print (accuracy1.mean()*100)

model1.fit(x_train, y_train)
predicted1 = model1.predict(x_test)
p1 = confusion_matrix(y_test, predicted1)
cm_display1 = metrics.ConfusionMatrixDisplay(confusion_matrix = p1, display_labels = [False, True])
cm_display1.plot()

print ("The accuracy of KNN is : {:.4f} %".format(accuracy_score(y_test, predicted1)*100))
print ("The precision of KNN is : {:.4f} %".format(metrics.precision_score(y_test, predicted1)*100))
print ("The recall of KNN is : {:.4f} %".format(metrics.recall_score(y_test, predicted1)*100))
print ("The f1-score of KNN is : {:.4f} %".format(metrics.f1_score(y_test, predicted1)*100))

model2 = SVC()
accuracy2 = cross_val_score(model2, x, y, scoring='accuracy', cv = 10)
print (accuracy2.mean()*100)

model2.fit(x_train, y_train)

predicted2 = model2.predict(x_test)
p2=confusion_matrix(y_test, predicted2)
cm_display2 = metrics.ConfusionMatrixDisplay(confusion_matrix = p2, display_labels = [False, True])
cm_display2.plot()

print ("The accuracy of SVM is : {:.4f} %".format(accuracy_score(y_test, predicted2)*100))
print ("The precision of SVM is : {:.4f} %".format(metrics.precision_score(y_test, predicted2)*100))
print ("The recall of SVM is : {:.4f} %".format(metrics.recall_score(y_test, predicted2)*100))
print ("The f1-score of SVM is : {:.4f} %".format(metrics.f1_score(y_test, predicted2)*100))

model3 = LogisticRegression()
accuracy3 = cross_val_score(model3, x, y, scoring='accuracy', cv = 10)
print (accuracy3.mean()*100)

model3.fit(x_train, y_train)

predicted3 = model3.predict(x_test)
p3= confusion_matrix(y_test, predicted3)
cm_display3 = metrics.ConfusionMatrixDisplay(confusion_matrix = p3, display_labels = [False, True])
cm_display3.plot()


print ("The accuracy of Logistic Regression is : {:.4f} %".format(accuracy_score(y_test, predicted3)*100))
print ("The precision of Logistic Regression is : {:.4f} %".format(metrics.precision_score(y_test, predicted3)*100))
print ("The recall of Logistic Regression is : {:.4f} %".format(metrics.recall_score(y_test, predicted3)*100))
print ("The f1-score of Logistic Regression is : {:.4f} %".format(metrics.f1_score(y_test, predicted3)*100))

