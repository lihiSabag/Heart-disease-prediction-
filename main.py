# loading dataset
import pandas as pd
import numpy as np

# data preprocessing
from sklearn.preprocessing import StandardScaler

# data splitting
from sklearn.model_selection import train_test_split

# data modeling
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# All Model`s
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

# loading the csv data to a Pandas DataFrame
dataset = pd.read_csv("heart.csv")
print('\n#########################        Analyzing The Data        #########################\n\n')


print('\n============================ first 5 rows of the dataset ============================\n')
print(dataset.head(5))
print('\n=====================================================================================\n')

print('\n=========================== Getting some info about the data ========================\n')
dataset.info()
print('\n=====================================================================================\n')

print('\n============================ Checking for missing values ============================\n')
print(dataset.isnull().sum())
print('\n=====================================================================================\n')

print('\n==================== Checking the distribution of Target Variable ===================\n')
print(dataset['target'].value_counts())

print('\n=====================================================================================\n')

print('\n================== Find how many unique values object features have ==================\n')
for col in dataset.select_dtypes(include=[np.number]).columns:
  print(f"{col} has {dataset[col].nunique()} unique value")

print('\n=====================================================================================\n')

print('\n============================== Statistical measures =================================\n')
print(dataset.describe())
print('\n=====================================================================================\n')

print('\n=========================== Correlation among variables =============================\n')
print(dataset.corr())
print('\n=====================================================================================\n')

print('\n###########################        Data Processing        ###########################\n\n')
# Split into categorical variables and numeric variables
categorical = []
numerical = []
for column in dataset.columns:
    if len(dataset[column].unique()) <= 10:
        categorical.append(column)
    else:
        numerical.append(column)
categorical .remove('target')
print(f'Numerical Columns:  {dataset[numerical].columns}')
print('\n')
print(f'Categorical Columns: {dataset[categorical].columns}')

# convert some categorical variables into dummy variables
df = pd.get_dummies(dataset, columns=categorical)

print('Dummy Variables Operation \n', df.columns)

# normalize the range of independent variables or features of data
s_sc = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[col_to_scale] = s_sc.fit_transform(df[col_to_scale])
print('Feature scaling (Normalization) \n', df.columns)
print(df.head(5))


print('\n###########################        Model preparation        ###########################\n\n')

# Train / Test and Split
X = df.drop('target', axis=1)
y = df.target

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.15, random_state=120)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=120)


def print_score(clf, X_train, y_train, X_test, y_test,test_size, train=True):
  if train:
    pred = clf.predict(X_train)
    clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
    print("Train split: {}%".format(100-test_size))
    print("Train Result:\n================================================")
    print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

  elif train == False:
    pred = clf.predict(X_test)
    clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
    print("Test split: {}%".format( test_size))
    print("Test Result:\n================================================")
    print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


print('\n#########################        Support Vector machine        #########################\n\n')

svm_clf1 = SVC(gamma=0.1, C=1.0, kernel="rbf", probability=True, max_iter=-1)
svm_clf1.fit(X_train1, y_train1)
svm_clf2 = SVC(gamma=0.1, C=1.0, kernel="rbf", probability=True, max_iter=-1)
svm_clf2.fit(X_train2, y_train2)

SVM_model1 = SVC(gamma=0.1, C=1.0, kernel="linear", probability=True, max_iter=-1).fit(X_train1, y_train1)
SVM_model2 = SVC(gamma=0.1, C=1.0, kernel="linear", probability=True, max_iter=-1).fit(X_train2, y_train2)


print("========== SVM MODEL WITH kernel=rbf =============")
print_score(svm_clf1, X_train1, y_train1, X_test1, y_test1, 15,  train=True)
print_score(svm_clf1, X_train1, y_train1, X_test1, y_test1, 15,  train=False)

print_score(svm_clf2, X_train2, y_train2, X_test2, y_test2, 30,  train=True)
print_score(svm_clf2, X_train2, y_train2, X_test2, y_test2, 30, train=False)


print("\n========== SVM MODEL WITH kernel=linear =============\n")
print_score(SVM_model1, X_train1, y_train1, X_test1, y_test1, 15, train=True)
print_score(SVM_model1, X_train1, y_train1, X_test1, y_test1, 15, train=False)

print_score(SVM_model2, X_train2, y_train2, X_test2, y_test2, 30, train=True)
print_score(SVM_model2, X_train2, y_train2, X_test2, y_test2, 30, train=False)

print("\n========== BEST SVM MODEL - ROC Curve =============\n")

y_probabilities = svm_clf2.predict_proba(X_test2)[:, 1]
false_positive_rate_svm_clf, true_positive_rate_svm_clf, threshold_svm_clf = roc_curve(y_test2, y_probabilities)
print(roc_auc_score(y_test2, y_probabilities))

test_score = accuracy_score(y_test2, svm_clf2.predict(X_test2)) * 100
train_score = accuracy_score(y_train2, svm_clf2.predict(X_train2)) * 100
results_df = pd.DataFrame(data=[["Support Vector Machine", train_score, test_score]],
                          columns=['Model',  'Training Accuracy %','Testing Accuracy %'])
print(results_df)

print('\n#########################        Decision Tree Classifier        #########################\n\n')

dt_entropy1 = DecisionTreeClassifier(criterion='entropy', random_state=120, min_impurity_decrease=0.01)
dt_entropy1.fit(X_train1, y_train1)

dt_entropy2 = DecisionTreeClassifier(criterion='entropy', random_state=120, min_impurity_decrease=0.01)
dt_entropy2.fit(X_train2, y_train2)

dt_gini1 = DecisionTreeClassifier(criterion='gini', random_state=120, min_impurity_decrease=0.01)
dt_gini1.fit(X_train1, y_train1)

dt_gini2 = DecisionTreeClassifier(criterion='gini', random_state=120, min_impurity_decrease=0.01)
dt_gini2.fit(X_train2, y_train2)

print("========== Decision Tree with Entropy =============")

print_score(dt_entropy1, X_train1, y_train1, X_test1, y_test1, 15, train=True)
print_score(dt_entropy1, X_train1, y_train1, X_test1, y_test1, 15, train=False)

print_score(dt_entropy2, X_train2, y_train2, X_test2,  y_test2, 30, train=True)
print_score(dt_entropy2, X_train2, y_train2, X_test2,  y_test2, 30, train=False)

print("========== Decision Tree with Gini =============")

print_score(dt_gini1, X_train1, y_train1, X_test1,y_test1, 15, train=True)
print_score(dt_gini1, X_train1, y_train1, X_test1,y_test1, 15, train=False)

print_score(dt_gini2, X_train2, y_train2, X_test2, y_test2, 30, train=True)
print_score(dt_gini2, X_train2, y_train2, X_test2, y_test2, 30, train=False)


print("\n========== BEST Decision Tree MODEL - ROC Curve =============\n")

y_probabilities = dt_gini2.predict_proba(X_test2)[:, 1]
false_positive_rate_tree_clf, true_positive_rate_tree_clf, threshold_tree_clf = roc_curve(y_test2, y_probabilities)
print(roc_auc_score(y_test2, y_probabilities))
test_score = accuracy_score(y_test2, dt_gini2.predict(X_test2)) * 100
train_score = accuracy_score(y_train2, dt_gini2.predict(X_train2)) * 100
results_df2 = pd.DataFrame(data=[["DT with gini", train_score, test_score]],
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df2, ignore_index=True)
print(results_df)
