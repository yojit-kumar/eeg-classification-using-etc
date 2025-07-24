import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def data_split(dataframe):
    volunteers = dataframe['volunteer'].unique()
    X = []
    y = []

    for v in volunteers:
        subdf = dataframe[dataframe['volunteer'] == v]

        etc_1 = subdf['ETC_EyesOpen']
        etc_2 = subdf['ETC_EyesClosed']

        #This if clause is specific to this dataset as all eyes closed state are processed after eyes open state.
        if len(etc_1) == 64 and len(etc_2) == 64:
            X.append(etc_1)
            y.append([0])
            X.append(etc_2)
            y.append([1])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    X_train_norm = (X_train-np.min(X_train,0))/(np.max(X_train,0)-np.min(X_train,0))
    X_train_norm = X_train_norm.astype(float)
    X_test_norm = (X_test-np.min(X_test,0))/(np.max(X_test,0)-np.min(X_test,0))
    X_test_norm = X_test_norm.astype(float)

    return X_train_norm, y_train, X_test_norm, y_test




def adaboost_train(X_train, y_train):
    n_estimator = [50, 100, 200, 300]
    BESTF1 = 0
    FOLD_NO = 5

    KF = KFold(n_splits=FOLD_NO, random_state=42, shuffle=True)
    KF.get_n_splits(X_train)

    for NEST in n_estimator:
        FSCORE_TEMP = []

        for train_idx, val_idx in KF.split(X_train):
            X_TRAIN, X_VAL = X_train[train_idx], X_train[val_idx]
            Y_TRAIN, Y_VAL = y_train[train_idx], y_train[val_idx]

            clf = AdaBoostClassifier(n_estimators=NEST, random_state=42)
            clf.fit(X_TRAIN, Y_TRAIN.ravel())
            Y_PRED = clf.predict(X_VAL)

            f1 = f1_score(Y_VAL, Y_PRED, average='macro')
            FSCORE_TEMP.append(f1)

        MEAN_FSCORE_TEMP = np.mean(FSCORE_TEMP)
        if MEAN_FSCORE_TEMP > BESTF1:
            BESTF1 = MEAN_FSCORE_TEMP
            BESTNEST = NEST

    RESULT_PATH = '../results/parameters/adaboost/'

    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print("Creation of result dir not required")

    np.save(RESULT_PATH+"NEST.npy", np.array([BESTNEST]))
    np.save(RESULT_PATH+"F1SCORE.npy", np.array([BESTF1]))

    print("Training Finished!")


def adaboost_test(X_train, y_train, X_test, y_test):
    RESULT_PATH = '../results/parameters/adaboost/'

    NEST = np.load(RESULT_PATH+'NEST.npy')[0]
    F1SCORE = np.load(RESULT_PATH+'F1SCORE.npy')[0]

    clf = AdaBoostClassifier(n_estimators=NEST, random_state=42)
    clf.fit(X_train, y_train.ravel())
    
    Y_TEST = y_test
    Y_PRED = clf.predict(X_test)

    acc = accuracy_score(Y_TEST, Y_PRED)
    f1 = f1_score(Y_TEST, Y_PRED, average='macro')
    prec = precision_score(Y_TEST, Y_PRED, average='macro')
    recall = recall_score(Y_TEST, Y_PRED, average='macro')


    print('TRAINING F1 Score', F1SCORE)

    print('ACCURACY', acc)
    print('TESTING F1 Score', f1)
    print('PRECISION', prec)
    print('RECALL', recall)

    np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
    np.save(RESULT_PATH+"/ACCURACY_TEST.npy", np.array([acc]) )
    np.save(RESULT_PATH+"/PRECISION_TEST.npy", np.array([prec]) )
    np.save(RESULT_PATH+"/RECALL_TEST.npy", np.array([recall]) )

    
def decisiontree_train(X_train, y_train):
    msl_values = [2, 5, 10, 15]
    md_values = [2, 3, 4, 5]

    BESTF1 = 0
    FOLD_NO = 5

    KF = KFold(n_splits=FOLD_NO, random_state=42, shuffle=True)
    KF.get_n_splits(X_train)

    clf = DecisionTreeClassifier(random_state=42)
    ccp_path = clf.cost_complexity_pruning_path(X_train, y_train)
    alpha = ccp_path['ccp_alphas']

    for MSL in msl_values:
        for MD in md_values:
            for CCP in alpha:
                FSCORE_TEMP = []
                clf = DecisionTreeClassifier(min_samples_leaf=MSL, random_state=42, max_depth=MD, ccp_alpha=CCP)
                for train_idx, val_idx in KF.split(X_train):
                    X_TRAIN, X_VAL = X_train[train_idx], X_train[val_idx]
                    Y_TRAIN, Y_VAL = y_train[train_idx], y_train[val_idx]

                    clf.fit(X_TRAIN, Y_TRAIN.ravel())
                    Y_PRED = clf.predict(X_VAL)

                    f1 = f1_score(Y_VAL, Y_PRED, average='macro')
                    FSCORE_TEMP.append(f1)

                MEAN_FSCORE_TEMP = np.mean(FSCORE_TEMP)
                if MEAN_FSCORE_TEMP > BESTF1:
                    BESTF1 = MEAN_FSCORE_TEMP
                    BESTMSL = MSL
                    BESTMD = MD
                    BESTCCP = CCP


    RESULT_PATH = '../results/parameters/decisiontree/'

    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print("Creation of result dir not required")

    np.save(RESULT_PATH+"MSL.npy", np.array([BESTMSL]))
    np.save(RESULT_PATH+"MD.npy", np.array([BESTMD]))
    np.save(RESULT_PATH+"CCP.npy", np.array([BESTCCP]))
    np.save(RESULT_PATH+"F1SCORE.npy", np.array([BESTF1]))

    print("Training Finished!")


def decisiontree_test(X_train, y_train, X_test, y_test):
    RESULT_PATH = '../results/parameters/decisiontree/'

    MSL = np.load(RESULT_PATH+'MSL.npy')[0]
    MD = np.load(RESULT_PATH+'MD.npy')[0]
    CCP = np.load(RESULT_PATH+'CCP.npy')[0]
    F1SCORE = np.load(RESULT_PATH+'F1SCORE.npy')[0]

    clf = DecisionTreeClassifier(min_samples_leaf=MSL, random_state=42, max_depth=MD, ccp_alpha=CCP)
    clf.fit(X_train, y_train.ravel())
    
    Y_TEST = y_test
    Y_PRED = clf.predict(X_test)

    acc = accuracy_score(Y_TEST, Y_PRED)
    f1 = f1_score(Y_TEST, Y_PRED, average='macro')
    prec = precision_score(Y_TEST, Y_PRED, average='macro')
    recall = recall_score(Y_TEST, Y_PRED, average='macro')


    print('TRAINING F1 Score', F1SCORE)

    print('ACCURACY', acc)
    print('TESTING F1 Score', f1)
    print('PRECISION', prec)
    print('RECALL', recall)

    np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
    np.save(RESULT_PATH+"/ACCURACY_TEST.npy", np.array([acc]) )
    np.save(RESULT_PATH+"/PRECISION_TEST.npy", np.array([prec]) )
    np.save(RESULT_PATH+"/RECALL_TEST.npy", np.array([recall]) )




def gaussiannb(X_test, y_test, X_train, y_train):
    RESULT_PATH = '../results/parameters/gnb/'

    clf = GaussianNB()
    clf.fit(X_train, y_train.ravel())

    Y_TEST = y_test
    Y_PRED = clf.predict(X_test)

    acc = accuracy_score(Y_TEST, Y_PRED)
    f1 = f1_score(Y_TEST, Y_PRED, average='macro')
    prec = precision_score(Y_TEST, Y_PRED, average='macro')
    recall = recall_score(Y_TEST, Y_PRED, average='macro')

    print('ACCURACY', acc)
    print('TESTING F1 Score', f1)
    print('PRECISION', prec)
    print('RECALL', recall)

    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print("Creation of result dir not required")

    np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
    np.save(RESULT_PATH+"/ACCURACY_TEST.npy", np.array([acc]) )
    np.save(RESULT_PATH+"/PRECISION_TEST.npy", np.array([prec]) )
    np.save(RESULT_PATH+"/RECALL_TEST.npy", np.array([recall]) )
    



def knn_train(X_train, y_train):
    k_values = [1, 3, 5, 7, 9, 11]

    BESTF1 = 0
    FOLD_NO = 5

    KF = KFold(n_splits=FOLD_NO, random_state=42, shuffle=True)
    KF.get_n_splits(X_train)

    for K in k_values:
        FSCORE_TEMP = []
        clf = KNeighborsClassifier(n_neighbors = K)
        for train_idx, val_idx in KF.split(X_train):
            X_TRAIN, X_VAL = X_train(train_idx), X_train(val_idx)
            Y_TRAIN, Y_VAL = y_train(train_idx), y_train(val_idx)

            clf.fit(X_TRAIN, Y_TRAIN.ravel())
            Y_PRED = clf.predict(X_VAL)
            
            f1 = f1_score(Y_VAL, Y_PRED, average='macro')
            FSCORE_TEMP.append(f1)
        
        MEAN_FSCORE_TEMP = np.mean(FSCORE_TEMP)
        if MEAN_FSCORE_TEMP > BESTF1:
            BESTF1 = MEAN_FSCORE_TEMP
            BESTK = K

    RESULT_PATH = '../results/parameters/kNN/'

    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print("Creation of result dir not required")

    np.save(RESULT_PATH+"K.npy", np.array([BESTK]))
    np.save(RESULT_PATH+"F1SCORE_TEST.npy", np.array([BESTF1]))

    print('Training Finished !')


def knn_test(X_train, y_train, X_test, y_test):
    RESULT_PATH = '../results/parameters/kNN/'

    K = np.load(RESULT_PATH+"K.npy")[0]
    F1SCORE = np.load(RESULT_PATH+"F1SCORE.npy")[0]

    clf = KNeighborsClassifier(n_neighbours=K)
    clf.fit(X_train, y_train)

    Y_TEST = y_test
    Y_PRED = clf.predict(X_test)

    acc = accuracy_score(Y_TEST, Y_PRED)
    f1 = f1_score(Y_TEST, Y_PRED, average='macro')
    prec = precision_score(Y_TEST, Y_PRED, average='macro')
    recall = recall_score(Y_TEST, Y_PRED, average='macro')


    print('TRAINING F1 Score', F1SCORE)

    print('ACCURACY', acc)
    print('TESTING F1 Score', f1)
    print('PRECISION', prec)
    print('RECALL', recall)

    np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
    np.save(RESULT_PATH+"/ACCURACY_TEST.npy", np.array([acc]) )
    np.save(RESULT_PATH+"/PRECISION_TEST.npy", np.array([prec]) )
    np.save(RESULT_PATH+"/RECALL_TEST.npy", np.array([recall]) )


def logisticr_train(X_train, y_train):
    C_values = [0.001, 0.01, 0.1, 10, 100, 1000]

    BESTF1 = 0
    FOLD_NO= 5
    
    KF = KFold(n_splits=FOLD_NO, random_state=42, shuffle=True)

    for C in C_values:
        FSCORE_TEMP=[]
        clf = LogisticRegression(penalty='l1', C=C, random_state=42, solver='liblinear')
        for train_idx, val_idx in KF.split(X_train):
            X_TRAIN, X_VAL = X_train(train_idx), X_train(val_idx)
            Y_TRAIN, Y_VAL = y_train(train_idx), y_train(val_idx)

            clf.fit(X_TRAIN, Y_TRAIN.ravel())
            Y_PRED = clf.predict(X_VAL)
            
            f1 = f1_score(Y_VAL, Y_PRED, average='macro')
            FSCORE_TEMP.append(f1)
        
        MEAN_FSCORE_TEMP = np.mean(FSCORE_TEMP)
        if MEAN_FSCORE_TEMP > BESTF1:
            BESTF1 = MEAN_FSCORE_TEMP
            BESTC = C

    RESULT_PATH = '../results/parameters/LR/'

    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print("Creation of result dir not required")

    np.save(RESULT_PATH+"C.npy", np.array([BESTC]))
    np.save(RESULT_PATH+"F1SCORE_TEST.npy", np.array([BESTF1]))

    print('Training Finished !')


def logisticr_test(X_train, y_train, X_test, y_test):
    RESULT_PATH = '../results/parameters/LR/'

    C = np.load(RESULT_PATH+"C.npy")[0]
    F1SCORE = np.load(RESULT_PATH+"F1SCORE.npy")[0]

    
    clf = LogisticRegression(penalty='l1', C=C, random_state=42, solver='liblinear')
    clf.fit(X_train, y_train)

    Y_TEST = y_test
    Y_PRED = clf.predict(X_test)

    acc = accuracy_score(Y_TEST, Y_PRED)
    f1 = f1_score(Y_TEST, Y_PRED, average='macro')
    prec = precision_score(Y_TEST, Y_PRED, average='macro')
    recall = recall_score(Y_TEST, Y_PRED, average='macro')


    print('TRAINING F1 Score', F1SCORE)

    print('ACCURACY', acc)
    print('TESTING F1 Score', f1)
    print('PRECISION', prec)
    print('RECALL', recall)

    np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
    np.save(RESULT_PATH+"/ACCURACY_TEST.npy", np.array([acc]) )
    np.save(RESULT_PATH+"/PRECISION_TEST.npy", np.array([prec]) )
    np.save(RESULT_PATH+"/RECALL_TEST.npy", np.array([recall]) )



def randomforest_train(X_train, y_train):
    n_estimators = [100, 200, 300, 500]
    md_values = [3, 5, 10, None]
    msl_values = [1,2,4]

    BESTF1 = 0
    FOLD_NO = 5
    
    KF = KFold(n_splits=FOLD_NO, random_state=42, shuffle=True)

    for NEST in n_estimators:
        for MD in md_values:
            for MSL in msl_values:
                FSCORE_TEMP = []
                clf = RandomForestClassifier(n_estimators=NEST, max_depth=MD, min_samples_leaf=MSL, random_state=42)
                for train_idx, val_idx in KF.split(X_train):
                    X_TRAIN, X_VAL = X_train[train_idx], X_train[val_idx]
                    Y_TRAIN, Y_VAL = y_train[train_idx], y_train[val_idx]

                    clf.fit(X_TRAIN, Y_TRAIN.ravel())
                    Y_PRED = clf.predict(X_VAL)

                    f1 = f1_score(Y_VAL, Y_PRED, average='macro')
                    FSCORE_TEMP.append(f1)

                MEAN_FSCORE_TEMP = np.mean(FSCORE_TEMP)
                if MEAN_FSCORE_TEMP > BESTF1:
                    BESTF1 = MEAN_FSCORE_TEMP
                    BESTMSL = MSL
                    BESTMD = MD
                    BESTNEST = NEST


    RESULT_PATH = '../results/parameters/randomforest/'

    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print("Creation of result dir not required")

    np.save(RESULT_PATH+"MSL.npy", np.array([BESTMSL]))
    np.save(RESULT_PATH+"MD.npy", np.array([BESTMD]))
    np.save(RESULT_PATH+"NEST.npy", np.array([BESTNEST]))
    np.save(RESULT_PATH+"F1SCORE.npy", np.array([BESTF1]))

    print("Training Finished!")


def randomforest_test(X_train, y_train, X_test, y_test):
    RESULT_PATH = '../results/parameters/randomforest/'

    NEST = np.load(RESULT_PATH+"NEST.npy")[0]
    MD = np.load(RESULT_PATH+"MD.npy")[0]
    MSL = np.load(RESULT_PATH+"MSL.npy")[0]
    F1SCORE = np.load(RESULT_PATH+"F1SCORE.npy")[0]

    
    clf = RandomForestClassifier(n_estimators=NEST, max_depth=MD, min_samples_leaf=MSL, random_state=42)
    clf.fit(X_train, y_train)

    Y_TEST = y_test
    Y_PRED = clf.predict(X_test)

    acc = accuracy_score(Y_TEST, Y_PRED)
    f1 = f1_score(Y_TEST, Y_PRED, average='macro')
    prec = precision_score(Y_TEST, Y_PRED, average='macro')
    recall = recall_score(Y_TEST, Y_PRED, average='macro')


    print('TRAINING F1 Score', F1SCORE)

    print('ACCURACY', acc)
    print('TESTING F1 Score', f1)
    print('PRECISION', prec)
    print('RECALL', recall)

    np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
    np.save(RESULT_PATH+"/ACCURACY_TEST.npy", np.array([acc]) )
    np.save(RESULT_PATH+"/PRECISION_TEST.npy", np.array([prec]) )
    np.save(RESULT_PATH+"/RECALL_TEST.npy", np.array([recall]) )




def svm_train(X_train, y_train):
    C_values = [0.1, 1, 10, 100]

    BESTF1 = 0
    FOLD_NO= 5
    
    KF = KFold(n_splits=FOLD_NO, random_state=42, shuffle=True)

    for C in C_values:
        FSCORE_TEMP=[]
        clf = SVC(C=C, random_state=42)
        for train_idx, val_idx in KF.split(X_train):
            X_TRAIN, X_VAL = X_train(train_idx), X_train(val_idx)
            Y_TRAIN, Y_VAL = y_train(train_idx), y_train(val_idx)

            clf.fit(X_TRAIN, Y_TRAIN.ravel())
            Y_PRED = clf.predict(X_VAL)
            
            f1 = f1_score(Y_VAL, Y_PRED, average='macro')
            FSCORE_TEMP.append(f1)
        
        MEAN_FSCORE_TEMP = np.mean(FSCORE_TEMP)
        if MEAN_FSCORE_TEMP > BESTF1:
            BESTF1 = MEAN_FSCORE_TEMP
            BESTC = C

    RESULT_PATH = '../results/parameters/SVM/'

    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print("Creation of result dir not required")

    np.save(RESULT_PATH+"C.npy", np.array([BESTC]))
    np.save(RESULT_PATH+"F1SCORE_TEST.npy", np.array([BESTF1]))

    print('Training Finished !')


def svm_test(X_train, y_train, X_test, y_test):
    RESULT_PATH = '../results/parameters/SVM/'

    C = np.load(RESULT_PATH+"C.npy")[0]
    F1SCORE = np.load(RESULT_PATH+"F1SCORE.npy")[0]

    
    clf = SVC(C=C, random_state=42)
    clf.fit(X_train, y_train)

    Y_TEST = y_test
    Y_PRED = clf.predict(X_test)

    acc = accuracy_score(Y_TEST, Y_PRED)
    f1 = f1_score(Y_TEST, Y_PRED, average='macro')
    prec = precision_score(Y_TEST, Y_PRED, average='macro')
    recall = recall_score(Y_TEST, Y_PRED, average='macro')


    print('TRAINING F1 Score', F1SCORE)

    print('ACCURACY', acc)
    print('TESTING F1 Score', f1)
    print('PRECISION', prec)
    print('RECALL', recall)

    np.save(RESULT_PATH+"/F1SCORE_TEST.npy", np.array([f1]) )
    np.save(RESULT_PATH+"/ACCURACY_TEST.npy", np.array([acc]) )
    np.save(RESULT_PATH+"/PRECISION_TEST.npy", np.array([prec]) )
    np.save(RESULT_PATH+"/RECALL_TEST.npy", np.array([recall]) )
