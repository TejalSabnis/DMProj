__author__ = 'Tejal'

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier

data = []
target = []

f = open("C:\\Users\\Tejal\\Documents\\ASU\\2_Spring 2016\\CSE 572 Data Mining\\Project\\Scikit\\train_data(1).txt",'r')
for line in f:
    linestrlist = line.strip().split(',')
    linenumlist = []
    for str in linestrlist:
        if '.' in str:
            linenumlist.append(float(str))
        else:
            linenumlist.append(int(str))
    data.append(linenumlist)
f.close()

f = open("C:\\Users\\Tejal\\Documents\\ASU\\2_Spring 2016\\CSE 572 Data Mining\\Project\\Scikit\\train_label(1).txt",'r')
for line in f:
    target.append(int(line.strip()))
f.close()

# print data
# print target

# Standardizing the data
max_abs_scaler = preprocessing.MaxAbsScaler()
data_maxabs = max_abs_scaler.fit_transform(data)

# create a base classifier used to evaluate a subset of attributes
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# model.fit(data, target)
# cv = cross_validation.ShuffleSplit(920, n_iter=10, test_size=0.1, random_state=0)
# scores = cross_validation.cross_val_score(model, data_maxabs, target, cv=cv)
# print "GradientBoostingClassifier "
# print scores.mean()

# create the RFE model and select 15 attributes
rfe = RFE(model, 15)
rfe = rfe.fit(data_maxabs, target)

# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)