from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.problem_transform import BinaryRelevance
#from keras.models import Sequential
#from keras.layers import Dense


#class KerasDNNWrapper(object):
#
#    def __init__(self, epochs=50, batch_size=10, Ni=5001):
#        # https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
#
#        self.__model = Sequential()
#        self.__model.add(Dense(int(Ni / 3), input_dim=Ni, activation='relu'))
#        self.__model.add(Dense(int(Ni / 10), activation='relu'))
#        self.__model.add(Dense(int(Ni / 100), activation='relu'))
#        self.__model.add(Dense(int(Ni / 500), activation='relu'))
#        self.__model.add(Dense(1, activation='sigmoid'))
#
#        self.__model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#        self.__model.summary()
#
#        self.__epochs = epochs
#        self.__batch_size = batch_size
#
#    def fit(self, X, Y):
#        self.__model.fit(X, Y, epochs=self.__epochs, batch_size=self.__batch_size, verbose=2)
#        return self
#
#    def predict(self, X):
#        predict = self.__model.predict(X)
#        return [1 if x > 0.5 else 0 for x in predict]


class MLFactory(object):
    @staticmethod
    def factory(type):
        if type == "AdaBoostClassifier":
            return BinaryRelevance(AdaBoostClassifier())
        elif type == "AdaBoostClassifier_s":
            return AdaBoostClassifier()
        elif type == "DecisionTreeClassifier":
            return DecisionTreeClassifier()
        elif type == "ExtraTreesClassifier":
            return ExtraTreesClassifier()
        elif type == "GaussianNB":
            return BinaryRelevance(GaussianNB())
        elif type == "GaussianNB_s":
            return GaussianNB()
        elif type == "GradientBoosting_42":
            return GradientBoostingClassifier(subsample=0.5, n_estimators=1000, learning_rate=0.01, random_state=42)
        elif type == "KNeighborsClassifier":
            return KNeighborsClassifier()
        elif type == "MultinomialNB":
            return BinaryRelevance(MultinomialNB())
        elif type == "MultinomialNB_s":
            return MultinomialNB()
        elif type == "QuadraticDiscriminantAnalysis":
            return BinaryRelevance(QuadraticDiscriminantAnalysis())
        elif type == "QuadraticDiscriminantAnalysis_s":
            return QuadraticDiscriminantAnalysis()
        elif type == "RandomForestClassifier":
            return RandomForestClassifier(n_estimators=80, min_samples_leaf=1, verbose=2)
        elif type == "RandomForestClassifier_104":
            return RandomForestClassifier(n_estimators=104, min_samples_leaf=1, verbose=2)
        elif type == "RandomForestClassifier_128":
            return RandomForestClassifier(n_estimators=128, min_samples_leaf=1, verbose=2)
        elif type == "RandomForestClassifier_256":
            return RandomForestClassifier(n_estimators=256, min_samples_leaf=1)
        elif type == "RandomForestClassifier_deterministic":
            return RandomForestClassifier(n_estimators=80, min_samples_leaf=1, random_state=42)
        elif type == "RandomForestClassifier_depth":
            return RandomForestClassifier(max_depth=5)
        elif type == "SGDClassifier":
            return BinaryRelevance(SGDClassifier(penalty='elasticnet', loss='log', verbose=2))
        elif type == "SGDClassifier_s":
            return SGDClassifier(penalty='elasticnet', loss='log', verbose=2)
        elif type == "SVCClassifier":
            return BinaryRelevance(SVC(kernel="linear", C=0.025))
        elif type == "SVCClassifier_s":
            return SVC(kernel="linear", C=0.025, probability=True)
#        elif type == "dnn":
#            return KerasDNNWrapper()
        elif type == "LogisticRegression":
            return LogisticRegression(solver='saga', multi_class='auto', random_state=42)
        elif type == "LogisticRegression_balanced":
            return LogisticRegression(C=0.1, solver='saga', multi_class='auto')
        else:
            assert 0, "Bad algorithm type: " + type