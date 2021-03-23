
def load_data(path: str) -> arr:
    import pandas as pd

    # path = "data/"

    test_df = pd.read_csv(path + "test.csv")
    train_df = pd.read_csv(path + "train.csv")
    data = [train_df, test_df]
    return data



def data_processing(data: arr) -> arr:
    for dataset in data:
        dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
        dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
        dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
        dataset['not_alone'] = dataset['not_alone'].astype(int)
    # train_df['not_alone'].value_counts()
    data[0] = data[0].drop(['PassengerId'], axis=1)
    return data



def data_cleaning(data: arr) -> arr:
    import re
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

    for dataset in data:
        dataset['Cabin'] = dataset['Cabin'].fillna("U0")
        dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(0)
        dataset['Deck'] = dataset['Deck'].astype(int)
        
    data[0] = data[0].drop(['Cabin'], axis=1)
    data[1] = data[1].drop(['Cabin'], axis=1)
    return data



def data_cleaning2(data: arr) -> pandas.core.frame.DataFrame:
    import numpy as np

    for dataset in data:
        mean = data[0]["Age"].mean()
        std = data[1]["Age"].std()
        is_null = dataset["Age"].isnull().sum()
        
        rand_age = np.random.randint(mean - std, mean + std, size = is_null)
        
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        dataset["Age"] = age_slice
        dataset["Age"] = data[0]["Age"].astype(int)

    common_value = 'S'
    for dataset in data:
        dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

    for dataset in data:
        dataset['Fare'] = dataset['Fare'].fillna(0)
        dataset['Fare'] = dataset['Fare'].astype(int)

    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in data:
        # extract titles
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        # replace titles with a more common title or as Rare
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        # convert titles into numbers
        dataset['Title'] = dataset['Title'].map(titles)
        # filling NaN with 0, to get safe
        dataset['Title'] = dataset['Title'].fillna(0)
    data[0] = data[0].drop(['Name'], axis=1)
    data[1] = data[1].drop(['Name'], axis=1)

    genders = {"male": 0, "female": 1}
    for dataset in data:
        dataset['Sex'] = dataset['Sex'].map(genders)

    data[0] = data[0].drop(['Ticket'], axis=1)
    data[1] = data[1].drop(['Ticket'], axis=1)

    ports = {"S": 0, "C": 1, "Q": 2}
    for dataset in data:
        dataset['Embarked'] = dataset['Embarked'].map(ports)


    for dataset in data:
        dataset['Age'] = dataset['Age'].astype(int)
        dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
        dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


    for dataset in data:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
        dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
        dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
        dataset['Fare'] = dataset['Fare'].astype(int)


    for dataset in data:
        dataset['Age_Class']= dataset['Age']* dataset['Pclass']


    for dataset in data:
        dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
        dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

    train_df = data[1]
    return train_df



def clean_data(train_df: pandas.core.frame.DataFrame) -> arr:
    train_labels = train_df['Survived']
    train_df = train_df.drop('Survived', axis=1)

    train_data = [train_df, train_labels]
    return train_data



def random_forest1(train_data: arr) -> numpy.float64:

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(train_data[0], train_data[1])
    acc_random_forest = round(random_forest.score(train_data[0], train_data[1]) * 100, 2)
    return acc_random_forest



def logreg1(train_data: arr) -> numpy.float64:
    logreg = LogisticRegression(solver='lbfgs', max_iter=110)
    logreg.fit(train_data[0], train_data[1])
    acc_log = round(logreg.score(train_data[0], train_data[1]) * 100, 2)
    return acc_log



def gaussian1(train_data: arr) -> numpy.float64:
    gaussian = GaussianNB()
    gaussian.fit(train_data[0], train_data[1])
    acc_gaussian = round(gaussian.score(train_data[0], train_data[1]) * 100, 2)
    return acc_gaussian



def linear_svc1(train_data: arr) -> numpy.float64:
    linear_svc = SVC(gamma='auto')
    linear_svc.fit(train_data[0], train_data[1])
    acc_linear_svc = round(linear_svc.score(train_data[0], train_data[1]) * 100, 2)
    return acc_linear_svc



def decision_tree1(train_data: arr) -> numpy.float64:
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(train_data[0], train_data[1])
    acc_decision_tree = round(decision_tree.score(train_data[0], train_data[1]) * 100, 2)
    return acc_decision_tree



def results1(acc_linear_svc: numpy.float64, acc_log: numpy.float64, acc_random_forest: numpy.float64, acc_gaussian: numpy.float64, acc_decision_tree: numpy.float64) -> pandas.core.frame.DataFrame:
    results = pd.DataFrame({
        'Model': ['Support Vector Machines', 'logistic Regression',
                'Random Forest', 'Naive Bayes', 'Decision Tree'],
        'Score': [acc_linear_svc, acc_log,
                acc_random_forest, acc_gaussian, acc_decision_tree]})
    result_df = results.sort_values(by='Score', ascending=False)
    result_df = result_df.set_index('Score')
    return result_df

