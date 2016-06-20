__author__ = 'Tejal'

from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy

if __name__ == '__main__':

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

    # Standardizing the training data
    max_abs_scaler = preprocessing.MaxAbsScaler()
    data_maxabs = max_abs_scaler.fit_transform(data)

    print data_maxabs[0]

    index = [0,2,10,13]
    for list in data_maxabs:
        list = numpy.delete(list,index)

    print data_maxabs[0]


    # model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    # model.fit(data_maxabs, target)
    # cv = cross_validation.ShuffleSplit(920, n_iter=10, test_size=0.1, random_state=0)
    # scores = cross_validation.cross_val_score(model, data_maxabs, target, cv=cv)
    # print "GradientBoostingClassifier "
    # print scores.mean()

    # train model
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(data_maxabs, target)

    # model evaluation
    cv = cross_validation.ShuffleSplit(920, n_iter=10, test_size=0.1, random_state=0)
    scores = cross_validation.cross_val_score(model, data_maxabs, target, cv=cv)
    print "AdaBoostClassifier "
    print scores.mean()

    # model = RandomForestClassifier()
    # model.fit(data_maxabs, target)
    # cv = cross_validation.ShuffleSplit(920, n_iter=10, test_size=0.1, random_state=0)
    # scores = cross_validation.cross_val_score(model, data_maxabs, target, cv=cv)
    # print "RandomForestClassifier "
    # print scores.mean()



    # test data
    test = []
    f = open("C:\\Users\\Tejal\\Documents\\ASU\\2_Spring 2016\\CSE 572 Data Mining\\Project\\Scikit\\test_data(1).txt",'r')
    for line in f:
        linestrlist = line.strip().split(',')
        linenumlist = []
        for str in linestrlist:
            if '.' in str:
                linenumlist.append(float(str))
            else:
                linenumlist.append(int(str))
        test.append(linenumlist)
    f.close()

    # print(test)

    # Standardizing test data
    test_maxabs = max_abs_scaler.transform(test)

    # print(test_maxabs)

    # test model (predict labels)
    predicted = model.predict(test_maxabs)

    # print(predicted)
    # prints labels
    for num in predicted:
        print num