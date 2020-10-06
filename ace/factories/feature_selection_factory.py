from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2, SelectKBest, SelectFromModel, f_classif, RFE
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC


class FeatureSelectionFactory(object):
    def __init__(self, k=5000):
        self.k = k

    def get_model(self, type):
        if type == "X2":
            return SelectKBest(chi2, k=self.k)
        elif type == "lsvc":
            lsvc = LinearSVC(C=0.1, penalty="l1", dual=False)
            return SelectFromModel(lsvc)
        elif type == "ExtraTrees":
            clf = ExtraTreesClassifier(n_estimators=self.k)
            clf = clf
            return SelectFromModel(clf)
        elif type == "ANOVA":
            return SelectKBest(f_classif, k=self.k)
        elif type == "SGD":
            return RFE(estimator=SGDClassifier(), n_features_to_select=self.k, step=10000, verbose=5)
        elif type == "Logistic":
            return RFE(estimator=LogisticRegression(solver="sag"), n_features_to_select=self.k, step=10000, verbose=5)
        else:
            assert 0, "Bad feature selection type: " + type