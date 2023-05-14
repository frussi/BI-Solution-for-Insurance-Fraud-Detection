import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
import seaborn as sn
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import accuracy_score as acc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from sklearn.svm import SVC
import plotly.graph_objects as go
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV

# Import data
data = pd.read_csv('insurance_claims.csv')

# Print dataset head and summary statistics
head = data.head()
print(head.to_string())

desc = data.describe()
print(desc.to_string())

# Drop _c39 column, full of null values
data.drop('_c39',axis=1,inplace=True)

# Create correlation Matrix
sns.set(style="white")
f, ax = plt.subplots(figsize=(18, 12))

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data.corr(), cmap=cmap, vmax=.3, center=0,annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt = '.2g')
plt.show()

# Print unique values per column
print(data.nunique())

# Change fraud_reported collumn to int/bool
data['fraud_reported'] = data['fraud_reported'].str.replace('Y', '1')
data['fraud_reported'] = data['fraud_reported'].str.replace('N', '0')
data['fraud_reported'] = data['fraud_reported'].astype(int)

# Plot histograms of data separated by fraud reported
plots = []

def vis_data(data, x, y = 'fraud_reported', graph = 'countplot'):
    a = data.groupby([x, y]).count()
    a.reset_index(inplace=True)
    no_fraud = a[a['fraud_reported'] == 0]
    yes_fraud = a[a['fraud_reported'] == 1]
    trace1 = go.Bar(x=no_fraud[x], y=no_fraud['policy_number'], name='No Fraud')
    trace2 = go.Bar(x=yes_fraud[x], y=yes_fraud['policy_number'], name='Fraud')
    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(title='{x} vs. {y}'.format(x=x, y=y))
    fig.update_layout(barmode='group')
    plots.append(fig)

colList = data.columns.values.tolist()
colList.remove('fraud_reported')
plotnumber = len(colList)

for col in colList:
    if plotnumber >= 0:
        vis_data(data, col)

# Export plots to html
for i in range(len(plots)):
    fig1 = plots.pop()
    with open('plots.html', 'a') as f:
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))

# Change all values in insured_hobbies column to other if not chess or cross fit
hobbies = data['insured_hobbies'].unique()
for hobby in hobbies:
  if (hobby != 'chess') & (hobby != 'cross-fit'):
    data['insured_hobbies'] = data['insured_hobbies'].str.replace(hobby, 'other')

# Based on stats and plots, only use columns that are appropriate
columnsToKeep = ['insured_sex', 'insured_occupation',
       'insured_hobbies', 'capital-gains', 'capital-loss', 'incident_type', 'collision_type', 'incident_severity',
       'authorities_contacted', 'incident_hour_of_the_day', 'number_of_vehicles_involved', 'incident_state',
       'witnesses', 'total_claim_amount', 'fraud_reported', 'age', 'policy_annual_premium', 'insured_zip', 'auto_year']

data1 = data
data = data[columnsToKeep]

plt.figure(figsize = (25, 20))
plotnumber = 1

# Encode categorical columns
columns_to_encode = []
for col in data.columns:
  if data[col].dtype == 'object':
    columns_to_encode.append(col)

data2 = data
data = pd.get_dummies(data, columns = columns_to_encode)
print(data.info())

# Create list of features
features = []

for col in data.columns:
  if col != 'fraud_reported':
    features.append(col)

print(len(features))

# Density plots to check data quality
for col in data.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.distplot(data[col])
        plt.xlabel(col, fontsize=15)

    plotnumber += 1

plt.tight_layout()
plt.show()

# Box plots to check for outliers
plt.figure(figsize=(20, 15))
plotnumber = 1

for col in data.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.boxplot(data[col])
        plt.xlabel(col, fontsize=15)

    plotnumber += 1
plt.tight_layout()
plt.show()

# Split the dataset into training and testing sets
target = 'fraud_reported'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)

# Standardize data due to outliers
st_x= StandardScaler()
X_train= st_x.fit_transform(X_train)
X_test= st_x.transform(X_test)

features = X.columns.values.tolist()

model_preds = {}
# Create Decision Tree
print("Decision Tree:")

# Create Decision Tree Model
dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train,y_train)
y_train_pred = dtree.predict(X_train)
y_test_pred = dtree.predict(X_test)

# Print Accuracy of Decision Tree
print(f'Train Score {accuracy_score(y_train_pred,y_train)}')
print(f'Test Score {accuracy_score(y_test_pred,y_test)}')

# Plot Decision Tree
tree.plot_tree(dtree, feature_names=features)
plt.title("Decision Tree for Fraud Detection")
plt.show()

# Prune Decision Tree with cost complexity pruning
path = dtree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas.shape)

# Add model to a list for each alpha
dtrees = []
for ccp_alpha in ccp_alphas:
    dtree = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    dtree.fit(X_train, y_train)
    dtrees.append(dtree)

# Plot accuracy vs alpha graph and print
train_acc = []
test_acc = []
for c in dtrees:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred, y_test))

plt.scatter(ccp_alphas, train_acc)
plt.scatter(ccp_alphas, test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy', drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy', drawstyle="steps-post")
plt.legend()
plt.title('Decision Tree Accuracy vs Alpha')
plt.show()

d = {'alpha': ccp_alphas, 'acc': test_acc}
df = pd.DataFrame(data=d)
print(df)

# Create pruned decision tree model
dtree_ = DecisionTreeClassifier(random_state=0,ccp_alpha=0.004652)
dtree_.fit(X_train,y_train)
y_train_pred = dtree_.predict(X_train)
y_pred = dtree_.predict(X_test)

# Print Accuracy of pruned Decision Tree
print(f'Train Score {accuracy_score(y_train_pred,y_train)}')
print(f'Test Score {accuracy_score(y_pred,y_test)}')

# Print pruned Decision Tree
tree.plot_tree(dtree_, feature_names=features)
plt.title("Pruned Decision Tree for Fraud Detection")
plt.show()

# Print decision tree accuracy, metrics, and feature importance
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
print(dtree_.feature_importances_)

# Print decision tree confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Print AUC of decision tree
tpr, fpr, threshold = roc_curve(y_pred, y_test, pos_label=1)
model_preds["Decision Tree"] = [tpr, fpr]
print()
print("AUC value = "+str(auc(tpr, fpr)))

# Create SVC Model
svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
svc_train_acc = accuracy_score(y_train, svc.predict(X_train))
svc_test_acc = accuracy_score(y_test, y_pred)

# Print Accuracy of SVC Model
print(f"Training accuracy of Support Vector Classifier is : {svc_train_acc}")
print(f"Test accuracy of Support Vector Classifier is : {svc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Improve Model with Hyperparameter Tuning
degrees = [2,3,4,5,6,7,8]
kernels = ['poly', 'rbf', 'sigmoid']
c_value = [1,2,3]

scores = {}
for degree in degrees:
    for kernel in kernels:
        for c in c_value:
            svc_t = SVC(kernel=kernel, degree=degree, C=c)
            svc_t.fit(X_train, y_train)
            preds = svc_t.predict(X_test)
            score = svc_t.score(X_test, y_test)
            scores['Score with degree as {d}, kernel as {k}, C as {c} is best'.format(d=degree, k=kernel, c=c)] = score

print(max(scores, key=scores.get))

# Create improved SVC model
svc_tuned = SVC(kernel='rbf', degree = 2, C = 1)
svc_tuned.fit(X_train, y_train)

y_pred = svc_tuned.predict(X_test)

# Print svc accuracy, metrics
print('Score:' , svc_tuned.score(X_test, y_test))
print('Classification report:', classification_report(y_test, y_pred))

# Print improved SVC confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.title("SVC Confusion Matrix")
plt.show()

# Print SVC AUC
tpr, fpr, threshold = roc_curve(y_pred, y_test, pos_label=1)
model_preds["SVC"] = [tpr, fpr]
print()
print("AUC value = "+str(auc(tpr, fpr)))

# Create KNN model
print("KNN:")

# Create list of error rates for n = 1 - 40
error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i, metric='cosine')
 knn.fit(X_train, y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))

# Plot error rates vs neighbors
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('KNN Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
req_k_value = error_rate.index(min(error_rate)) + 1
print("Minimum error:-", min(error_rate), "at K =", req_k_value)
plt.show()

# Build and fit KNN model
knn = KNeighborsClassifier(n_neighbors = req_k_value, metric='cosine')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(knn)

# Print KNN accuracy, metrics
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Print KNN confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.title("KNN Confusion Matrix")
plt.show()

# Print KNN AUC
tpr, fpr, threshold = roc_curve(y_pred, y_test, pos_label=1)
model_preds["KNN"] = [tpr, fpr]
print()
print("AUC value = "+str(auc(tpr, fpr)))

# Create Gaussian Naive Bayes Model
print(len(features))
naiveB = GaussianNB()

# Create the forward sequential feature selection
sfs1 = sfs(naiveB,
               k_features= 12,
               forward=True,
               floating=True,
               verbose=2,
               scoring='accuracy',
               cv=5)

sfs1 = sfs1.fit(X_train, y_train)

# Print results of selected features
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)

# Build model using all features
naiveB = GaussianNB()
naiveB.fit(X_train, y_train)

# Print training and testing accuracy of all features
y_train_pred = naiveB.predict(X_train)
print('Training accuracy on all features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = naiveB.predict(X_test)
print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))

# Build model with selected features
naiveB = GaussianNB()
naiveB.fit(X_train[:, feat_cols], y_train)

# Print training and testing accuracy of selected features
y_train_pred = naiveB.predict(X_train[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = naiveB.predict(X_test[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))

y_pred = naiveB.predict(X_test[:, feat_cols])
print(naiveB)

# Print Naive Bayes accuracy, metrics
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
print(metrics.classification_report(y_test, y_pred))

# Print Naive Bayes confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# Print Naive Bayes AUC
tpr, fpr, threshold = roc_curve(y_pred, y_test, pos_label=1)
model_preds["Gaussian Naive Bayes"] = [tpr, fpr]
print()
print("AUC value = "+str(auc(tpr, fpr)))

# Print ROC Curve for all models
plt.title("ROC curve for various classifiers:")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

for key, value in model_preds.items():
    model_list = model_preds[key]
    plt.plot(model_list[0], model_list[1], label=key)
    plt.legend()
plt.show()
