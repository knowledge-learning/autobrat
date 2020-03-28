from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#Load datasets

#load training set
X_training_set = []
y_training_set = []
with open(Path(__file__).parent.parent / "data" / "autobrat" / "result") as fp:
    training_set = fp.read()

#load pool selectio set
X_pool_set = []
with open(Path(__file__).parent.parent / "data" / "autobrat" / "corpus") as fp:
    X_training_set = fp.read()

#Train classifier
classifier = SVC()
classifier.fit(X_training_set, y_training_set)

#metric
classifier.decision_function(X_pool_set)
#ordenar las palabras
#obtener palabra más relevante