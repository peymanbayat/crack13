import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image

st.set_page_config(page_title="Bariscan Machine Learning 📈", page_icon=":guardsman:")


st.title("BB Machine Leraning Stremlit")

st.write("""
## Explore Different Classifier
### which one is the best ?
""")

dataset_name  = st.sidebar.selectbox("Select Datset", ("Iris", "Breast Cancer", "Wine Dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('Number of classes:', len(np.unique(y)))

def add_parameter_ui(clsf_name):
    params = dict()
    if clsf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clsf_name =="SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else :
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clsf_name, params):
    if clsf_name == "KNN":
        clsf = KNeighborsClassifier(n_neighbors = params["K"])
    elif clsf_name =="SVM":
        clsf = SVC(C = params["C"])
    else :
        clsf = RandomForestClassifier(n_estimators = params["n_estimators"],
                                      max_depth = params["max_depth"],
                                      random_state = 1234)
    return clsf

clsf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clsf.fit(X_train, y_train)
y_pred = clsf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()


#plt.show()
st.pyplot(fig)

GitHub = "https://github.com/BariscanBilgen"
Linkedin = "https://www.linkedin.com/in/bariscanbilgen/"
Medium = "https://medium.com/@bariscanbilgen"
YouTube = "https://www.youtube.com/@bariscanbilgen"

col1, col2 = st.columns(2)

image = Image.open("bb.jpg")
col1.image(image, use_column_width=True)

col2.write("### Bariscan BILGEN")
col2.write("##### BI | Data Analyst	:bar_chart:")

col2.write(f"[GitHub]({GitHub})")
col2.write(f"[Linkedin]({Linkedin})")
col2.write(f"[Medium]({Medium})")
col2.write(f"[YouTube]({YouTube})")