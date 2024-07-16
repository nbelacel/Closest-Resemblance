
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def feature_selection(dataset,targets):
    data = pd.DataFrame()
    column = ["Column" + " " + str(i) for i in range(dataset.shape[1])]
    X = pd.DataFrame(data=dataset, columns=column)
    y = pd.DataFrame(data=targets, columns=["class"])
    for i in column:
        X[i] = X[i].fillna(0)
    clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

    clf.fit(X, y)
    selected={}
    for feature in zip(column, clf.feature_importances_):
        selected[feature[0]]=feature[1]


    sort_orders = sorted(selected.items(), key=lambda x: x[1], reverse=True)
    #print(sort_orders)
    th=0
    for i in sort_orders:
        if th <= 0.5:
            data[i[0]]=X[i[0]]
            th=th+i[1]
        else:
            data = data.to_numpy()
            print("Number of features selected ", data.shape[1])
            return data



    '''
    for i in [0.1,0.05,0.01,0.005,0.001]:
        for feature in zip(column, clf.feature_importances_):
            if feature[1] >= i:
                data[feature[0]]=X[feature[0]]
        if not(data.empty):
            data = data.to_numpy()
            print("Number of features selected ", data.shape[1])
            return data
        else:
            print("NO column for i = "+str(i))

'''

