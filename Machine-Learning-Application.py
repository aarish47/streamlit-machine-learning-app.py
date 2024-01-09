# import your libraries
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Heading for our app
st.write(''' 
# SmartWeb: Diverse Machine learning app!
### - Created by: Aarish Asif Khan
### - Held on: 9th January 2024
##### 
To utilize our app, simply pick a dataset and select a classifier of your choice. Our system will then generate results tailored to your selections.
''')

# Creating a sidebar on our web application to show the three main datasets
dataset_name = st.sidebar.selectbox(
    '#### Select a Dataset',
    ('Iris', 'Digits', 'Wine')
)

# Creating another sidebar on our web app to show the three classifiers which we will apply on them
classifier_name = st.sidebar.selectbox(
    '#### Select any Classifier',
    ('K-Nearest Neighbor', 'Select Vector Machine', 'Random Forest')
)

# Creating a function and fetching the three datasets
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Digits':
        data = datasets.load_digits()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target 
    return x, y
    
# Calling the function now
x, y = get_dataset(dataset_name)

# Printing the shape of our dataset 
st.write('- **Dataset Structure:**', x.shape)
st.write('- **Class Count:**', len(np.unique(y)))

# adding parameters to our classifier
def add_parameter_ui(classifier_name):
    params = dict() # creating an empty dictionary 
    if classifier_name =='K-Nearest Neighbor':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K 
    elif classifier_name == 'Select Vector Machine':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C 
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth 
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators # number of trees 
    return params 

# lets call the function
params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == "K-Nearest Neighbor":
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif classifier_name == "Select Vector Machine":
        clf = SVC(C=params['C'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                           max_depth=params['max_depth'], random_state=1234)
    return clf
 
if st.checkbox('Check the code!') :
    with st.echo():
        clf = get_classifier(classifier_name, params)
        # train test split 
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234) 
        # training the data 
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # checking the accuracy score
        acc = accuracy_score(y_test, y_pred)

# get the classifier
clf = get_classifier(classifier_name, params)

# train test split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234) 

# training the data 
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# checking the accuracy score
acc = accuracy_score(y_test, y_pred)
st.write(f'- **Classifier =** {classifier_name}')
st.write(f'- **Accuracy =** ', acc)

# making a plot 
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]


fig = plt.figure()
plt.scatter(x1, x2,
    c=y, alpha=0.8,
    cmap='viridis')

plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.colorbar()

# showing our figure
# plt.show()
st.pyplot(fig)
