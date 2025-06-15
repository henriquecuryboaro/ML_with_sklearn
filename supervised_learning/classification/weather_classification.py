import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 0. Declare useful functions for data analysis and machine learning

# 0.1. Define function to remove outliers based on interquartile range method
def data_cleaning(data,features):
    for feature in features:
        Q1 = data[feature].quantile(0.25) #primeiro quartil
        Q3 = data[feature].quantile(0.75) #terceiro quartil
        IQR = Q3 - Q1 #Amplitude interquartil
        data = data[(data[feature] > (Q1 - 1.5*IQR)) & (data[feature] < (Q3 + 1.5*IQR))]
    return data   

# 0.2. Define function to separate features by type
def TypeSeparation(data):
    num_cols = data.select_dtypes(include = ['int64', 'float64']).columns
    cat_cols = data.select_dtypes(exclude = ['int64', 'float64']).columns
    numeric_features, categorical_features = [column for column in num_cols], [column for column in cat_cols]
    return num_cols, cat_cols, numeric_features, categorical_features

# 0.3. Define function to tune a model's hyperparameters
def ParamGridSearch(model,params,data,target):
    clf = model
    param_grid = params
    grid_search = GridSearchCV(clf, 
                           param_grid, 
                           cv=5,
                           scoring='accuracy')
    grid_search.fit(data,target)
    print(f'Best parameters for {clf}: {grid_search.best_params_}')

# 0.4. Define function to train and evaluate a model based on its accuracy
def train_evaluate(model,data_train,target_train,data_test,target_test):
    model.fit(data_train,target_train)
    predicted = model.predict(data_test)
    named_steps = model.named_steps  
    print(f'Accuracy of {named_steps["classifier"]}: {round(accuracy_score(target_test,predicted),3)}')

# 0.5. Import dataset
dataset = pd.read_csv('weather_classification_data.csv')

# 0.7. Separate target from features data and verify if there are imbalanced classes in target
target = dataset['Weather Type']
data = dataset.drop(columns=['Weather Type'])
classes = target.unique()

# 0.6. separate features by type by calling apropriate function
num_cols, cat_cols, numeric_features, categorical_features = TypeSeparation(data)

# 1. Exploratory data analysis section

# 1.1. remove outliers from original dataset
clean_dataset = data_cleaning(dataset, numeric_features)

# 1.2. Plot correlations between numeric features
# num_data=clean_data[numeric_features]
# fig = px.imshow(num_data.corr(),text_auto=True)
# fig.show()

# 1.3. Plot boxplot of numeric features
# plt.figure(figsize=(12, 8))
# for i, column in enumerate(num_cols):
#     plt.subplot(4, 3, i+1)
#     sns.boxplot(y=num_data[column])
#     plt.title(f'Boxplot of {column}')
#     plt.tight_layout()

# plt.show()

# 2. Machine Learning section

# 2.1. Pass target as the parameter for stratify to obtain target_train and target_test arrays with the balance as the original dataset
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=1, stratify=target)

# 2.2. Define a pipeline for preprocessing features data
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore")),])

preprocessor=ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                             ("cat", categorical_transformer, categorical_features),])


# Tune hyperparameters of classifiers
param_grid_RF = {'n_estimators':[10,20,30,40,50,60],
             'max_depth':[3,5,7,9],
                'max_features':[1,2,3,4,5]}

ParamGridTree = {'max_depth':[1,2,3,4,5],
             'min_samples_split':[2,3,4,5],
             'min_samples_leaf':[1,2,3,4,5]}

#preprocess features to feed into GridSearchCV validation method
# data_prepared = preprocessor.fit_transform(data)

#Determine hyperparameters
# ParamGridSearch(RandomForestClassifier(),param_grid_RF,data_prepared,target)
# ParamGridSearch(DecisionTreeClassifier(),ParamGridTree,data_prepared,target)

# 2.4. Define full pipelines including the classification algorithms
RFclf = Pipeline(steps=[("preprocessor", preprocessor), 
                    ("classifier", RandomForestClassifier())])

DTreeClf = Pipeline(steps=[("preprocessor", preprocessor), 
                    ("classifier", DecisionTreeClassifier())])

RFclfTuned = Pipeline(steps=[("preprocessor", preprocessor), 
                    ("classifier", RandomForestClassifier(max_depth=9, max_features=5, n_estimators=50))])

DTreeClfTuned = Pipeline(steps=[("preprocessor", preprocessor), 
                    ("classifier", DecisionTreeClassifier(max_depth=5, min_samples_leaf=1, min_samples_split=2))])

# 2.6. Evaluate models by verifying its accuracy and analysing confusion matrix
models = [RFclf, DTreeClf, RFclfTuned, DTreeClfTuned] 

for model in models:
    model.fit(data_train, target_train)
    predictions = model.predict(data_test)
    cr = classification_report(target_test, predictions, target_names=classes)
    named_steps = model.named_steps
    train_evaluate(model,data_train,target_train,data_test,target_test)
    print(f'Classification report for {named_steps["classifier"]}')
    print(cr)

plt.figure(figsize=(16, 8))
for i, model in enumerate(models):
    plt.subplot(2, 2, i+1)
    model.fit(data_train, target_train)
    predictions = model.predict(data_test)
    cm = confusion_matrix(target_test, predictions)
    sns.heatmap(cm, annot=True,fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes) 
    named_steps = model.named_steps
    plt.title(f'Confusion matrix for \n {named_steps["classifier"]}')
plt.tight_layout()
plt.show()
