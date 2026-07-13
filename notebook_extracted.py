import pandas as pd
df = pd.read_csv("Dataset.csv")
df

df.isna().sum()

duplicates = df[df.duplicated(keep=False)]
print('Duplicate values in the Dataset=', duplicates)

df.info()

import plotly.express as px

fig = px.histogram(df, x='SepsisLabel', color='SepsisLabel',
                   color_discrete_map={0: 'lightpink', 1: 'lightblue'},
                   labels={'SepsisLabel': 'Sepsis Label'},
                   title='Distribution of Sepsis Label')
fig.update_xaxes(title_text='Sepsis Label')
fig.update_yaxes(title_text='Count')
fig.show()


# import plotly.express as px

# # Define numeric columns
# numeric_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

# # Create box plots for each numeric feature grouped by SepsisLabel
# for col in numeric_columns:
#     fig = px.box(df, x='SepsisLabel', y=col, title=f'{col} vs Sepsis Label',
#                  labels={'SepsisLabel': 'Sepsis Label', col: col})
#     fig.update_xaxes(title_text='Sepsis Label')
#     fig.update_yaxes(title_text=col)
#     fig.show()



# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:11:29 2024

@author: kograeme
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

"""
Notes: 

-The iterative imputation method gives two files, both scaled and unscaled.
-If using KNN method, only one KNN file given due to initial scaling.
"""

#--------------------"""General parameters"""--------------------------

# impute_method = 0 for iterative imputation, 1 for KNN
impute_method = 0
 
# scale = 1 to scale data, scale = 0 not to scale
# only makes a difference if using iterative imputation, KNN is automatically scaled
# Only affects output dataframe. Creates scaled and unscaled csv files regardless.
scale = 1

# PCA = 1 to perform basic PCA analysis on scaled data, PCA = 0 to skip it
do_PCA = 0

# transform = 1 to power transform O2Sat, transform = 0 otherwise
transform = 1

# plot_dist = 1 if you want to plot each feature distribution after imputation
plot_dist = 0

# save = 1 to save csv files. save = 0 not to
save = 0

#----------------------------------------------------------------------


#---------------"""Null value handling parameters"""-------------------

#The minimum initial data to null ratio of each column.
# E.g. 0.01 = select if column has at least 1% good data (1% is 15,000 rows)
# A higher value is stricter, yielding less data.
dratio_init = 0.01

# Maximum nulls allowed in a row before imputation.
# A lower max_row_nulls value is stricter, yielding less data.
max_row_nulls = 10

#The minimum filtered data to null ratio of each column.
# E.g. 0.6 = Select if column has at least 60% good data before imputation.
# A higher value is stricter, yielding less data.
dratio_filt = 0.6
#----------------------------------------------------------------------


#---------------------"""Imputation parameters"""---------------------

# KNN imputation neighbours
k = 5

# Iterative imputation neighbours
impit = 10
#------------------------------------------------------------------

def cv_imputation(df, imputed_df, n_splits=5):
    df=df.reset_index(drop=True)
    imputed_df=imputed_df.reset_index(drop=True)
    
    #Returns mae averaged over all folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = []

    for train_index, test_index in kf.split(df):

        # Original DataFrame with missing values
        original_df = df.loc[test_index]
        original_df[df.isna()] = imputed_df[df.isna()]

        # Calculating mae for each column
        mae_per_column = (original_df - imputed_df).abs().mean()

        # Calculate overall mean absolute error
        average_mae = mae_per_column.mean()
        mae_scores.append(average_mae)
    
    av_mae_tot = np.mean(mae_scores)
    return av_mae_tot

def auto_clean(df_in,impute_method=0,scale=1,do_PCA=1,plot_dist=1,
               dratio_init=0.01,max_row_nulls=10,dratio_filt=0.6,
               k=5, impit=10, save=1, transform = 1, template_df = None):

    """
    returns df_out, the cleaned dataframe
    returns .csv files of imputed data    
    """
    
    df = df_in.copy()
    
    if template_df is not None:
        #Keeping only the columns that template_df kept after cleaning
        df = df[template_df.columns]
        print("Using the columns from the template dataset")
        rnull = df.isnull().sum(axis=1)
        f_df = df[rnull <= max_row_nulls].copy()
        
    else:
        print("Performing column selection on template df")
        
        """# Removing uninteresting or non-independent features"""
        drops = ["Unit1","Unit2","Patient_ID","Hour", "Unnamed: 0", "ICULOS", "HospAdmTime"]
        df.drop(columns=drops, inplace=True)
        
        """# Filtering by missing in column"""
        dataratio = df.count() / (df.count() + df.isna().sum())
        lowdata = dataratio[dataratio < dratio_init].sort_values()    
        lowdata_labs = np.ravel(lowdata.index)
        print(f"Dropping the columns: {lowdata_labs}")
        drops = lowdata_labs
        df.drop(columns=drops, inplace=True)
        
        """# Filtering by missing in row"""
        
        #Keeping only the rows that have enough non-Nan predictor values of a certain threshold
        rnull = df.isnull().sum(axis=1)
        #Keeping only the rows missing less than the threshold value and storing in f_df
        f_df = df[rnull <= max_row_nulls].copy()
    
        f_dataratio = f_df.count() / (f_df.count() + f_df.isna().sum())
        f_lowdata = f_dataratio[f_dataratio < dratio_filt].sort_values()
        f_lowdata_labs = np.ravel(f_lowdata.index)
        print(f"Dropping the columns: {f_lowdata_labs}")
        drops = f_lowdata_labs
        f_df.drop(columns=drops, inplace=True)
        
        #Re-running after to get an even train-test split after dropping all the columns we don't want
        return f_df
    
    if transform == 1 and "O2Sat" in f_df.columns:
        pt=PowerTransformer(method="yeo-johnson")
        f_df["O2Sat"] = pt.fit_transform(f_df["O2Sat"].values.reshape(-1, 1))
    
    
    if impute_method == 0:
        """# Iterative Imputation"""
        
        impute_ratio = f_df.isna().sum().sum() / f_df.count().sum()
        print(f"Imputing {impute_ratio*100:.2f}% of the total data with iterative imputation")
        
        imputer = IterativeImputer(max_iter=impit, random_state=0)
        imp_data = imputer.fit_transform(f_df)
        imp_df = pd.DataFrame(imp_data, columns=f_df.columns)
        if save == 1:
            print("Saving csv file: 'Clean_Data.csv'")
            imp_df.to_csv("Clean_Data.csv", header=True,index=False)
        df_out = imp_df.copy()
        
        """# Data Scaling"""
        cat_cols = ["SepsisLabel", "Gender"]
        num_cols = [col for col in imp_df.columns if col not in cat_cols]
        scaler = StandardScaler()
        imp_scaled = imp_df.copy()
        imp_scaled[num_cols] = scaler.fit_transform(imp_df[num_cols])
        if save == 1:
            print("Saving csv file: 'Clean_Data_Scaled.csv'")
            imp_scaled.to_csv("Clean_Data_Scaled.csv", header=True,index=False)
        if scale == 1:
            df_out = imp_scaled.copy()
            
        """"Imputation Analysis"""
        cat_cols = ["SepsisLabel", "Gender"]
        num_cols = [col for col in f_df.columns if col not in cat_cols]
        scaler = StandardScaler()
        f_scaled = f_df.copy()
        f_scaled[num_cols] = scaler.fit_transform(f_df[num_cols][~f_df.isnull()])
        av_mae = cv_imputation(f_scaled,imp_scaled,n_splits=10)
        print("Average Imputation MAE:", av_mae)        
                    
    if impute_method == 1:
        """# KNN Imputation"""
        impute_ratio = f_df.isna().sum().sum() / f_df.count().sum()
        print(f"Imputing {impute_ratio*100:.2f}% of the total data with KNN")
        
        #Will Scale non-Nan first from f_df
        #Then use KNNimputer        
        cat_cols = ["SepsisLabel", "Gender"]
        num_cols = [col for col in f_df.columns if col not in cat_cols]
        scaler = StandardScaler()
        f_scaled = f_df.copy()
        f_scaled[num_cols] = scaler.fit_transform(f_df[num_cols][~f_df.isnull()])
    
        imputer = KNNImputer(n_neighbors=k)
        knn_data = imputer.fit_transform(f_scaled)
        knn_df = pd.DataFrame(knn_data, columns=f_scaled.columns)
        if save == 1:
            print("Saving csv file: 'Clean_Data_KNN.csv'")
            knn_df.to_csv("Clean_Data_KNN.csv", header=True,index=False)    
        df_out = knn_df.copy()
        
        """"Imputation Analysis"""    
        av_mae = cv_imputation(f_scaled,knn_df,n_splits=10)
        print("Average Imputation MAE:", av_mae)      
    
    """# PCA"""    
    if do_PCA == 1:
        if impute_method == 0 and scale == 0:
            raise ValueError("Must scale iteratively imputed data to use PCA")
            
        X_df = imp_scaled.drop(columns="SepsisLabel")
        pca = PCA(n_components=2)  # Chooseing 2 first for easy visualization
        principalComponents = pca.fit_transform(X_df)
        principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
        label = imp_scaled['SepsisLabel']
        merged_df = principalDf.merge(label, left_index=True, right_index=True)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PC1', y='PC2', data=merged_df, s=70, alpha=0.7, edgecolor='k', hue='SepsisLabel')
        plt.title('PCA Plot by Label')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='SepsisLabel')
        plt.show()
        
        def plot_pca_biplot(score, loadings, labels_for_coloring, labels=None, palette='viridis'):
            xs = score[:, 0]
            ys = score[:, 1]
            score_df = pd.DataFrame({'PC1': xs, 'PC2': ys, 'Label': labels_for_coloring})
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x='PC1', y='PC2', data=score_df, hue='Label', palette=palette, alpha=0.5, edgecolor='k')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            for i, (loading_x, loading_y) in enumerate(loadings):
                plt.arrow(0, 0, loading_x*max(xs), loading_y*max(ys),
                          color='r', width=0.0005, head_width=0.02)
                if labels is not None:
                    plt.text(loading_x*max(xs)*1.2, loading_y*max(ys)*1.2, labels[i], color='r')
            plt.grid(True)
            plt.legend(title='SepsisLabel')
            plt.show()

        loadings = pca.components_.T
        feature_names=X_df.columns
        labelname = imp_scaled['SepsisLabel']
        plot_pca_biplot(principalComponents, loadings, labels_for_coloring=labelname, labels=feature_names)
        
        """#### Looking at all PCA components"""
        pca = PCA()
        pca.fit(X_df)
    
        plt.figure(figsize=(10, 7))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
        plt.title('Explained Variance by Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()
        
    if plot_dist == 1:
        """#### Perhaps we should consider transforming O2Sat"""
        # Checking the distribution of each feature 
        
        cat_cols = ["SepsisLabel", "Gender"]
        num_cols = [col for col in df_out.columns if col not in cat_cols]
        for column in num_cols:
            plt.figure()
            sns.histplot(df_out[column], kde=True)#kernel density estimate (KDE) plot
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()        
    
    print(f"Shape of output df: {np.shape(df_out)}")
    print("\n")
    df_out = df_out.reset_index(drop=True)
    return df_out


def split_clean(df_in,impute_method=0,scale=1,do_PCA=0,plot_dist=0,
               dratio_init=0.01,max_row_nulls=10,dratio_filt=0.6,
               k=5, impit=10, transform=1, save=0):
    
    #First getting df after dropping columns
    drop_df= auto_clean(df_in, dratio_init=dratio_init, max_row_nulls=max_row_nulls, dratio_filt=dratio_filt,
                            template_df = None)
    
    #Splitting df into training and test, then performing imputations/transformations/scaling separately
    train_df, test_df = train_test_split(drop_df, test_size=0.2, random_state=42)
    
    print("\n")
    print("Manipulating training dataset")
    train_cleaned = auto_clean(train_df, impute_method=impute_method, scale=scale, do_PCA=do_PCA, plot_dist=plot_dist,
                             k=k, impit=impit, transform = transform, save=save, template_df=drop_df)
    
    print("Manipulating testing dataset")    
    test_cleaned = auto_clean(test_df, impute_method=impute_method, scale=scale, do_PCA=0, plot_dist=0,
                             k=k, impit=impit, save=0, transform=transform,template_df=drop_df)  
    
    return train_cleaned, test_cleaned
    
    

df_raw = pd.read_csv("Dataset.csv")

'''
Example:
    
df_out = auto_clean(df_raw,impute_method=impute_method,scale=scale,do_PCA=do_PCA,plot_dist=plot_dist,
               dratio_init=dratio_init,max_row_nulls=max_row_nulls,dratio_filt=dratio_filt,
               k=k, impit=impit)

'''

# =============================================================================
# #-----------------
# #Data exploration - Comment when not needed
# #Printing out to see df size of what different parameters give:
# test_row_nulls = [4,5,6,7,8,9,10]  
# test_drat_filt = [0.55,0.6,0.7,0.8]
# for i in test_row_nulls:
#     for j in test_drat_filt:
#         print(f"Using max_row_nulls = {i} and filtered dratio = {j}:")
#         auto_clean(df_raw,impute_method=0,scale=1,do_PCA=0,plot_dist=0,
#                        dratio_init=0.01,max_row_nulls=i,dratio_filt=j,
#                        k=5, impit=10, save=0)
# #---------------
# 
# =============================================================================

#Example:
train_df, test_df = split_clean(df_raw,impute_method=impute_method,scale=scale,do_PCA=do_PCA,plot_dist=plot_dist,
               dratio_init=dratio_init,max_row_nulls=max_row_nulls,dratio_filt=dratio_filt,
               k=k, impit=impit, transform = transform, save=save)

# def exploratory_plots(train_df,test_df):
#     tot_df = pd.concat([train_df, test_df], axis=0)
#     cat_cols = ["SepsisLabel", "Gender"]
#     num_cols = [col for col in tot_df.columns if col not in cat_cols]
#     tot_df_numerical = tot_df[num_cols]
    
#     print(tot_df_numerical)
    
    
#     plt.rcParams['figure.figsize'] = (30, 25)
#     # Adjust the layout to fit all the plots
#     n_rows = int(np.ceil(len(tot_df_numerical.columns) / 4))
#     tot_df_numerical.plot(kind='box',  subplots=True, layout=(n_rows, 4), sharex=False, sharey=False, fontsize=18)
#     plt.show()
    
    
#     plt.rcParams['figure.figsize']=(3,2)
#     for i in range(tot_df_numerical.shape[1]):
#       tot_df_numerical.hist(column=num_cols[i], bins=50)
#     plt.show()
    
#     cor_num = tot_df_numerical.corr()
#     plt.figure(figsize=(15,15))
#     plt.title('Correlation')
#     a = sns.heatmap(cor_num, square=True, annot=True, fmt='.2f', linecolor='white')
#     a.set_xticklabels(a.get_xticklabels(), rotation=90)
#     a.set_yticklabels(a.get_yticklabels(), rotation=30)
#     plt.show()
    
# exploratory_plots(train_df,test_df)


X_train = train_df.drop(columns='SepsisLabel')
y_train= train_df['SepsisLabel']
X_test= test_df.drop(columns='SepsisLabel')
y_test= test_df['SepsisLabel']

X_train

train_df.isna().sum()

test_df.isna().sum()

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming you have X_train already defined
vif_data = pd.DataFrame() 
vif_data["feature"] = X_train.columns 

# Calculating VIF for each feature using X_train
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                   for i in range(len(X_train.columns))] 

print(vif_data)

# X_train = train_df.drop(columns='SepsisLabel')
# y_train= train_df['SepsisLabel']
# X_test= test_df.drop(columns='SepsisLabel')
# y_test= test_df['SepsisLabel']

X_train = train_df.drop(columns=['SepsisLabel','Hct'])
y_train= train_df['SepsisLabel']
X_test= test_df.drop(columns=['SepsisLabel','Hct'])
y_test= test_df['SepsisLabel']

# X_train = train_df.drop(columns=['SepsisLabel','Hgb'])
# y_train= train_df['SepsisLabel']
# X_test= test_df.drop(columns=['SepsisLabel','Hgb'])
# y_test= test_df['SepsisLabel']

# X_train = train_df.drop(columns=['SepsisLabel','Hct','MAP'])
# y_train= train_df['SepsisLabel']
# X_test= test_df.drop(columns=['SepsisLabel','Hct','MAP'])
# y_test= test_df['SepsisLabel']
# X_train.info()

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming you have X_train already defined
vif_data = pd.DataFrame() 
vif_data["feature"] = X_train.columns 

# Calculating VIF for each feature using X_train
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                   for i in range(len(X_train.columns))] 

print(vif_data)

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier

clf = DummyClassifier(strategy= 'most_frequent').fit(X_train,y_train)
y_pred = clf.predict(X_test)

print('y actual : \n' +  str(y_test.value_counts()))

print('y predicted : \n' + str(pd.Series(y_pred).value_counts()))

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))

from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from pycaret.classification import *

# Step 2: Load your dataset
# Example: df = pd.read_csv('your_dataset.csv')

# Step 3: Set up the PyCaret environment for regression
exp = setup(data=train_df, target='SepsisLabel', session_id=123)

# Step 4: Compare multiple regression models
best_model = compare_models()

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

qda = QuadraticDiscriminantAnalysis()

qda.fit(X_train, y_train)
pred = qda.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,pred)))
print('Precision Score : ' + str(precision_score(y_test,pred)))
print('Recall Score : ' + str(recall_score(y_test,pred)))
print('F1 Score : ' + str(f1_score(y_test,pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,pred)))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix

model3 = DecisionTreeClassifier()
model3.fit(X_train,y_train)
y_pred = model3.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

model = GradientBoostingClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

ridge_classifier = RidgeClassifier(alpha=1.0, solver='auto', random_state=42)
ridge_classifier.fit(X_train, y_train)
y_pred = ridge_classifier.predict(X_test)


print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

model1 = RandomForestClassifier()
model1.fit(X_train,y_train)
y_pred = model1.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

svc = SVC()
svc.fit(X_train,y_train)
pred=svc.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Initialize the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)


mlp.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = mlp.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)


# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:11:29 2024

@author: kograeme
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN

from pycaret.classification import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix

from tqdm import tqdm


import sys
#sys.stdout = open("automated_cleaning_output_imputed_ver2.txt", "w")

"""
Notes: 

-The iterative imputation method gives two files, both scaled and unscaled.
-If using KNN method, only one KNN file given due to initial scaling.
"""

#--------------------"""General parameters"""--------------------------

# impute_method = 0 for iterative imputation, 1 for KNN, 2 for no imputation testing
impute_method = 0
 
# scale = 1 to scale data, scale = 0 not to scale
# only makes a difference if using iterative imputation, KNN is automatically scaled
# Only affects output dataframe. Creates scaled and unscaled csv files regardless.
scale = 1

# PCA = 1 to perform basic PCA analysis on scaled data, PCA = 0 to skip it
do_PCA = 0

# transform = 1 to power transform O2Sat, transform = 0 otherwise
transform = 0

# plot_dist = 1 if you want to plot each feature distribution after imputation
plot_dist = 0

# save = 1 to save csv files. save = 0 not to
save = 0

#----------------------------------------------------------------------


#---------------"""Null value handling parameters"""-------------------

#The minimum initial data to null ratio of each column.
# E.g. 0.01 = select if column has at least 1% good data (1% is 15,000 rows)
# A higher value is stricter, yielding less data.
dratio_init = 0.01

# Maximum nulls allowed in a row before imputation.
# A lower max_row_nulls value is stricter, yielding less data.
max_row_nulls = 10

#The minimum filtered data to null ratio of each column.
# E.g. 0.6 = Select if column has at least 60% good data before imputation.
# A higher value is stricter, yielding less data.
dratio_filt = 0.6
#----------------------------------------------------------------------


#---------------------"""Imputation parameters"""---------------------

# KNN imputation neighbours
k = 5

# Iterative imputation neighbours
impit = 10
#------------------------------------------------------------------

def cv_imputation(df, imputed_df, n_splits=5):
    df=df.reset_index(drop=True)
    imputed_df=imputed_df.reset_index(drop=True)
    
    #Returns mae averaged over all folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scores = []

    for train_index, test_index in kf.split(df):

        # Original DataFrame with missing values
        original_df = df.loc[test_index]
        original_df[df.isna()] = imputed_df[df.isna()]

        # Calculating mae for each column
        mae_per_column = (original_df - imputed_df).abs().mean()

        # Calculate overall mean absolute error
        average_mae = mae_per_column.mean()
        mae_scores.append(average_mae)
    
    av_mae_tot = np.mean(mae_scores)
    return av_mae_tot

def auto_clean(df_in,impute_method=0,scale=1,do_PCA=1,plot_dist=1,
               dratio_init=0.01,max_row_nulls=10,dratio_filt=0.6,
               k=5, impit=10, save=1, transform = 1, template_df = None):

    """
    returns df_out, the cleaned dataframe
    returns .csv files of imputed data    
    """
    
    df = df_in.copy()    
        
    
    if impute_method == 2 and template_df is not None:
        #Keeping only the columns that template_df kept after cleaning
        df = df[template_df.columns]
        print("Using the columns from the template dataset")
        df_out = df.dropna()
        
        if scale == 1:
            cat_cols = ["SepsisLabel", "Gender","Patient_ID"]
            num_cols = [col for col in df.columns if col not in cat_cols]
            scaler = StandardScaler()
            df_scaled = df.copy()
            df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
            df_out_scaled = df_scaled.copy()
            if save == 1:
                print("Saving csv file: 'Clean_Data_Scaled_NoImpute.csv'")
                df_out_scaled.to_csv("Clean_Data_Scaled_NoImpute.csv", header=True,index=False)
            df_out_scaled.drop(columns="Patient_ID",inplace=True)
            return df_out_scaled
        
        if save == 1:
            print("Saving csv file: 'Clean_Data_Unscaled_NoImpute.csv'")
            df_out.to_csv("Clean_Data_Unscaled_NoImpute.csv", header=True,index=False)
        df_out.drop(columns="Patient_ID",inplace=True)        
        return df_out
    
    
    if template_df is not None:
        #Keeping only the columns that template_df kept after cleaning
        df = df[template_df.columns]
        print("Using the columns from the template dataset")
        rnull = df.isnull().sum(axis=1)
        f_df = df[rnull <= max_row_nulls].copy()
        
    else:
        if impute_method != 2:
            print("Performing column selection on template df")
            
            """# Removing uninteresting or non-independent features"""
            drops = ["Unit1","Unit2", "Unnamed: 0", "Hour","HospAdmTime",]#"ICULOS"] 
            df.drop(columns=drops, inplace=True)
            
            """# Filtering by missing in column"""
            dataratio = df.count() / (df.count() + df.isna().sum())
            lowdata = dataratio[dataratio < dratio_init].sort_values()    
            lowdata_labs = np.ravel(lowdata.index)
            print(f"Dropping the columns: {lowdata_labs}")
            drops = lowdata_labs
            df.drop(columns=drops, inplace=True)
            
            #if len(df) == 0:
                #return None
            
            """# Filtering by missing in row"""
            #Keeping only the rows that have enough non-Nan predictor values of a certain threshold
            rnull = df.isnull().sum(axis=1)
            #Keeping only the rows missing less than the threshold value and storing in f_df
            f_df = df[rnull <= max_row_nulls].copy()
        
            f_dataratio = f_df.count() / (f_df.count() + f_df.isna().sum())
            f_lowdata = f_dataratio[f_dataratio < dratio_filt].sort_values()
            f_lowdata_labs = np.ravel(f_lowdata.index)
            print(f"Dropping the columns: {f_lowdata_labs}")
            drops = f_lowdata_labs
            f_df.drop(columns=drops, inplace=True)
            return f_df
    
    if transform == 1 and "O2Sat" in f_df.columns:
        pt=PowerTransformer(method="yeo-johnson")
        cat_cols = ["SepsisLabel", "Gender","Patient_ID"]
        num_cols = [col for col in f_df.columns if col not in cat_cols]
            
        #f_df["O2Sat"] = pt.fit_transform(f_df["O2Sat"].values.reshape(-1, 1))
        #f_df["Hour"] = pt.fit_transform(f_df["Hour"].values.reshape(-1, 1))
        #f_df["ICULOS"] = pt.fit_transform(f_df["ICULOS"].values.reshape(-1, 1))
        #f_df["HospAdmTime"] = pt.fit_transform(f_df["HospAdmTime"].values.reshape(-1, 1))
    
    
    if impute_method == 0:
        """# Iterative Imputation"""
        
        impute_ratio = f_df.isna().sum().sum() / f_df.count().sum()
        print(f"Imputing {impute_ratio*100:.2f}% of the total data with iterative imputation")
        
        grouped_by_patient = f_df.groupby('Patient_ID')
        
        imputer = IterativeImputer(max_iter=impit, random_state=42, n_nearest_features=12, keep_empty_features = True)        
        imp_df = pd.DataFrame()
                
        for patient_id, group_data in grouped_by_patient:
            # Apply iterative imputation to the group data (excluding 'Patient_ID' column)
            patient_data = group_data.copy()
            patient_data.drop(columns=["Patient_ID"],inplace=True)
            imputed_group_data = pd.DataFrame(imputer.fit_transform(patient_data),columns=patient_data.columns)
           
            imputed_group_data['Patient_ID'] = patient_id
            imp_df = pd.concat([imp_df, imputed_group_data])
            
        # Reset the index of the imputed data
        imp_df.reset_index(drop=True, inplace=True)        
        
        #imputer = IterativeImputer(max_iter=impit, random_state=0)
        #imp_df = imputer.fit_transform(f_df)
        #imp_df = pd.DataFrame(imp_df, columns=f_df.columns)
        if save == 1:
            print("Saving csv file: 'Clean_Data.csv'")
            imp_df.to_csv("Clean_Data.csv", header=True,index=False)
        df_out = imp_df.copy()
        
        """# Data Scaling"""
        cat_cols = ["SepsisLabel", "Gender","Patient_ID"]
        num_cols = [col for col in imp_df.columns if col not in cat_cols]
        scaler = StandardScaler()
        imp_scaled = imp_df.copy()
        imp_scaled[num_cols] = scaler.fit_transform(imp_df[num_cols])
        if save == 1:
            print("Saving csv file: 'Clean_Data_Scaled.csv'")
            imp_scaled.to_csv("Clean_Data_Scaled.csv", header=True,index=False)
        if scale == 1:
            df_out = imp_scaled.copy()
            
        """"Imputation Analysis"""
        cat_cols = ["SepsisLabel", "Gender","Patient_ID"]
        num_cols = [col for col in f_df.columns if col not in cat_cols]
        scaler = StandardScaler()
        f_scaled = f_df.copy()
        f_scaled[num_cols] = scaler.fit_transform(f_df[num_cols][~f_df.isnull()])
        av_mae = cv_imputation(f_scaled,imp_scaled,n_splits=5)
        print("Average Imputation MAE:", av_mae)        
                    
    if impute_method == 1:
        """# KNN Imputation"""
        impute_ratio = f_df.isna().sum().sum() / f_df.count().sum()
        print(f"Imputing {impute_ratio*100:.2f}% of the total data with KNN")
        
        #Will Scale non-Nan first from f_df
        #Then use KNNimputer        
        cat_cols = ["SepsisLabel", "Gender","Patient_ID"]
        num_cols = [col for col in f_df.columns if col not in cat_cols]
        scaler = StandardScaler()
        f_scaled = f_df.copy()
        f_scaled[num_cols] = scaler.fit_transform(f_df[num_cols][~f_df.isnull()])
    
        imputer = KNNImputer(n_neighbors=k, keep_empty_features = True)
        knn_df = pd.DataFrame()
        
        grouped_by_patient = f_scaled.groupby('Patient_ID')
                
        for patient_id, group_data in grouped_by_patient:
            # Apply iterative imputation to the group data (excluding 'Patient_ID' column)
            patient_data = group_data.copy()
            patient_data.drop(columns=["Patient_ID"],inplace=True)
            imputed_group_data = pd.DataFrame(imputer.fit_transform(patient_data),columns=patient_data.columns)
            imputed_group_data['Patient_ID'] = patient_id
            knn_df = pd.concat([knn_df, imputed_group_data])
            
        # Reset the index of the imputed data
        knn_df.reset_index(drop=True, inplace=True)             
        #knn_data = imputer.fit_transform(f_scaled)
        #knn_df = pd.DataFrame(knn_data, columns=f_scaled.columns)
        
        if save == 1:
            print("Saving csv file: 'Clean_Data_KNN.csv'")
            knn_df.to_csv("Clean_Data_KNN.csv", header=True,index=False)    
        df_out = knn_df.copy()
        
        """"Imputation Analysis"""    
        av_mae = cv_imputation(f_scaled,knn_df,n_splits=5)
        print("Average Imputation MAE:", av_mae)
    
    if impute_method == 2 and template_df is None:
        print("Skipping imputation - Instead dropping sequentially")
        """Ignoring imputation for smaller dataset"""
        resulting_df = pd.DataFrame()
        while resulting_df.empty:
            temp_df = df.copy()
            
            """# Filtering by missing in column"""
            dataratio = temp_df.count() / (temp_df.count() + temp_df.isna().sum())
            lowdata = dataratio.sort_values()    
            
            drops = np.ravel(dataratio.sort_values().index)[0]
            print(f"Dropping the column: {drops}")
            temp_df.drop(columns=drops, inplace=True)
            
            temp_df = temp_df.dropna()
            print(np.shape(temp_df))
            resulting_df = temp_df

        df_out = df[resulting_df.columns]   
            
    
    """# PCA"""    
    if do_PCA == 1:
        if impute_method == 0 and scale == 0:
            raise ValueError("Must scale iteratively imputed data to use PCA")
            
        X_df = imp_scaled.drop(columns="SepsisLabel")
        pca = PCA(n_components=2)  # Choosing 2 first for easy visualization
        principalComponents = pca.fit_transform(X_df)
        principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
        label = imp_scaled['SepsisLabel']
        merged_df = principalDf.merge(label, left_index=True, right_index=True)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='PC1', y='PC2', data=merged_df, s=70, alpha=0.7, edgecolor='k', hue='SepsisLabel')
        plt.title('PCA Plot by Label')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='SepsisLabel')
        plt.show()
        
        def plot_pca_biplot(score, loadings, labels_for_coloring, labels=None, palette='viridis'):
            xs = score[:, 0]
            ys = score[:, 1]
            score_df = pd.DataFrame({'PC1': xs, 'PC2': ys, 'Label': labels_for_coloring})
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x='PC1', y='PC2', data=score_df, hue='Label', palette=palette, alpha=0.5, edgecolor='k')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            for i, (loading_x, loading_y) in enumerate(loadings):
                plt.arrow(0, 0, loading_x*max(xs), loading_y*max(ys),
                          color='r', width=0.0005, head_width=0.02)
                if labels is not None:
                    plt.text(loading_x*max(xs)*1.2, loading_y*max(ys)*1.2, labels[i], color='r')
            plt.grid(True)
            plt.legend(title='SepsisLabel')
            plt.show()

        loadings = pca.components_.T
        feature_names=X_df.columns
        labelname = imp_scaled['SepsisLabel']
        plot_pca_biplot(principalComponents, loadings, labels_for_coloring=labelname, labels=feature_names)
        
        """#### Looking at all PCA components"""
        pca = PCA()
        pca.fit(X_df)
    
        plt.figure(figsize=(10, 7))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
        plt.title('Explained Variance by Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()
        
    if plot_dist == 1:
        """#### Perhaps we should consider transforming O2Sat"""
        # Checking the distribution of each feature 
        
        cat_cols = ["SepsisLabel", "Gender"]
        num_cols = [col for col in df_out.columns if col not in cat_cols]
        for column in num_cols:
            plt.figure()
            sns.histplot(df_out[column], kde=True)#kernel density estimate (KDE) plot
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()        
    
    print(f"Shape of output df: {np.shape(df_out)}")
    print("\n")
    df_out = df_out.reset_index(drop=True)
    return df_out


def split_clean(df_in,impute_method=0,scale=1,do_PCA=0,plot_dist=0,
               dratio_init=0.01,max_row_nulls=10,dratio_filt=0.6,
               k=5, impit=10, transform=1, save=0):
    
    #First getting df after dropping columns
    drop_df= auto_clean(df_in, dratio_init=dratio_init, max_row_nulls=max_row_nulls, dratio_filt=dratio_filt,
                            template_df = None)
    
    #Splitting df into training and test, then performing imputations/transformations/scaling separately
    if len(drop_df) < 100:
        return pd.DataFrame(), pd.DataFrame()
    
    #For impute_method = 2, drop all Na from input df
    if impute_method == 2:
        drop_df = drop_df.dropna()
        
    train_df, test_df = train_test_split(drop_df, test_size=0.1, random_state=42)
    
    print("\n")
    print("Manipulating training dataset")
    train_cleaned = auto_clean(train_df, impute_method=impute_method, scale=scale, do_PCA=do_PCA, plot_dist=plot_dist,
                             k=k, impit=impit, transform = transform, save=save, template_df=drop_df)
    
    print("Manipulating testing dataset")    
    test_cleaned = auto_clean(test_df, impute_method=impute_method, scale=scale, do_PCA=0, plot_dist=0,
                             k=k, impit=impit, save=0, transform=transform,template_df=drop_df)  
    
    return train_cleaned, test_cleaned
    

def exploratory_plots(train_df,test_df):
    tot_df = pd.concat([train_df, test_df], axis=0)
    cat_cols = ["SepsisLabel", "Gender"]
    num_cols = [col for col in tot_df.columns if col not in cat_cols]
    tot_df_numerical = tot_df[num_cols]
    
    print(tot_df_numerical)
    
    
    plt.rcParams['figure.figsize'] = (30, 25)
    # Adjust the layout to fit all the plots
    n_rows = int(np.ceil(len(tot_df_numerical.columns) / 4))
    tot_df_numerical.plot(kind='box',  subplots=True, layout=(n_rows, 4), sharex=False, sharey=False, fontsize=18)
    plt.show()
    
    
    plt.rcParams['figure.figsize']=(3,2)
    for i in range(tot_df_numerical.shape[1]):
      tot_df_numerical.hist(column=num_cols[i], bins=50)
    plt.show()
    
    cor_num = tot_df_numerical.corr()
    plt.figure(figsize=(15,15))
    plt.title('Correlation')
    a = sns.heatmap(cor_num, square=True, annot=True, fmt='.2f', linecolor='white')
    a.set_xticklabels(a.get_xticklabels(), rotation=90)
    a.set_yticklabels(a.get_yticklabels(), rotation=30)
    plt.show()


def class_balancing(train_df, ratio=0.5):
        X_train = train_df.drop(columns='SepsisLabel')
        y_train= train_df['SepsisLabel']
        #oversample = SMOTE(sampling_strategy=0.2, random_state=42)
        #oversample = ADASYN(sampling_strategy=0.1, random_state=42)
        #oversample = RandomOverSampler(sampling_strategy=0.1, random_state=42)
        #X_resampled, y_resampled = oversample.fit_resample(X_train, y_train)
        undersample = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
        #X_resampled, y_resampled = undersample.fit_resample(X_resampled, y_resampled)
        X_resampled, y_resampled = undersample.fit_resample(X_train, y_train)
        
        #Putting the resampled data back into a dataframe
        train_df = pd.DataFrame(X_resampled, columns=X_train.columns)
        train_df['SepsisLabel'] = y_resampled
        return train_df
    

def modelling(train_df,test_df):
    X_train = train_df.drop(columns='SepsisLabel')
    y_train= train_df['SepsisLabel']
    X_test= test_df.drop(columns='SepsisLabel')
    y_test= test_df['SepsisLabel']
    
    model = RandomForestClassifier(n_estimators=300)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
    print('Precision Score : ' + str(precision_score(y_test,y_pred,zero_division=0)))
    print('Recall Score : ' + str(recall_score(y_test,y_pred)))
    print('F1 Score : ' + str(f1_score(y_test,y_pred)))
    print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))    
    print("\n")
    print("-----------------------------------------")
    print("\n")
    return model


df_raw = pd.read_csv("Dataset.csv")

'''
#Example:
    
df_out = auto_clean(df_raw,impute_method=impute_method,scale=scale,do_PCA=do_PCA,plot_dist=plot_dist,
               dratio_init=dratio_init,max_row_nulls=max_row_nulls,dratio_filt=dratio_filt,
               k=k, impit=impit)

'''

'''
#Example:
train_df, test_df = split_clean(df_raw,impute_method=impute_method,scale=scale,do_PCA=do_PCA,plot_dist=plot_dist,
               dratio_init=dratio_init,max_row_nulls=max_row_nulls,dratio_filt=dratio_filt,
               k=k, impit=impit, transform = transform, save=save)
'''
    
#-----------------
#Data exploration - Comment when not needed
test_imputes = [0]
test_drat_init = [0.01]
test_row_nulls = [11]  
test_drat_filt= [0.6]
test_sample_ratios = [0.75]

total_iterations = len(test_imputes) * len(test_drat_init) * len(test_row_nulls) * len(test_drat_filt) * len(test_sample_ratios)
progress_bar = tqdm(total=total_iterations, desc="Processing")

models = []
for r in test_sample_ratios:
    for m in test_imputes:
        for n in test_drat_init:
            for i in test_row_nulls:
                for j in test_drat_filt:
                    print(f"Using class ratio = {r}, max_row_nulls = {i}, dratio_filt = {j}, impute_method={m}, test_drat_init={n}:")
                    train_df, test_df = split_clean(df_raw,impute_method=m,scale=1,do_PCA=0,plot_dist=0,
                                   dratio_init=n,max_row_nulls=i,dratio_filt=j,
                                   k=5, impit=10, transform=1, save=0)
                    
                    if (len(train_df) < 1000)  or (len(test_df) < 1000):
                        print("Empty or small dataframes after cleaning")
                        print("\n")
                        continue
                    
                    #Major class imbalance.
                    #Resampling 
                    train_df = class_balancing(train_df, ratio=r)
                    
                    sepsis = len(train_df[train_df['SepsisLabel'] == 1])
                    non_sepsis = len(train_df[train_df['SepsisLabel'] == 0])
                    print(f'number of sepsis label 1 is {sepsis}')
                    print(f'while number of sepsis label 0 is {non_sepsis}')
                    print("train_df shape after sampling:",np.shape(train_df))
                    print("test_df shape:",np.shape(test_df))
                    print("\n")
                    
                    model = modelling(train_df,test_df)
                    if model in models:
                        progress_bar.update(1)
                        continue
                    
                    models.append(model)
                    
                    #Can use PyCaret to try and find the best model here         
                    # Set up the PyCaret environment for regression
                    #exp = setup(data=train_df, target='SepsisLabel', session_id=123)
                    #best_model = compare_models()
                    #print(best_model)
                    
                    progress_bar.update(1)
                
                                
progress_bar.close()                
                
#---------------

    


X_train = train_df.drop(columns='SepsisLabel')
y_train= train_df['SepsisLabel']
X_test= test_df.drop(columns='SepsisLabel')
y_test= test_df['SepsisLabel']
X_train

train_df.isna().sum()

test_df.isna().sum()

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming you have X_train already defined
vif_data = pd.DataFrame() 
vif_data["feature"] = X_train.columns 

# Calculating VIF for each feature using X_train
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                   for i in range(len(X_train.columns))] 

print(vif_data)

# X_train = train_df.drop(columns='SepsisLabel')
# y_train= train_df['SepsisLabel']
# X_test= test_df.drop(columns='SepsisLabel')
# y_test= test_df['SepsisLabel']

# X_train = train_df.drop(columns=['SepsisLabel','Hct'])
# y_train= train_df['SepsisLabel']
# X_test= test_df.drop(columns=['SepsisLabel','Hct'])
# y_test= test_df['SepsisLabel']

X_train = train_df.drop(columns=['SepsisLabel','Hgb'])
y_train= train_df['SepsisLabel']
X_test= test_df.drop(columns=['SepsisLabel','Hgb'])
y_test= test_df['SepsisLabel']


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming you have X_train already defined
vif_data = pd.DataFrame() 
vif_data["feature"] = X_train.columns 

# Calculating VIF for each feature using X_train
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                   for i in range(len(X_train.columns))] 

print(vif_data)

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier

clf = DummyClassifier(strategy= 'most_frequent').fit(X_train,y_train)
y_pred = clf.predict(X_test)

print('y actual : \n' +  str(y_test.value_counts()))

print('y predicted : \n' + str(pd.Series(y_pred).value_counts()))

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))

from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from pycaret.classification import *

# Step 2: Load your dataset
# Example: df = pd.read_csv('your_dataset.csv')

# Step 3: Set up the PyCaret environment for regression
exp = setup(data=train_df, target='SepsisLabel', session_id=123)

# Step 4: Compare multiple regression models
best_model = compare_models()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix

model3 = DecisionTreeClassifier()
model3.fit(X_train,y_train)
y_pred = model3.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define the decision tree classifier
model = DecisionTreeClassifier()

# Define the parameter grid to search
param_grid = {
    'criterion': ['gini', 'entropy','log_loss'],
    'max_depth': [5,6,7,8,9,11],
    'min_samples_split': [2,3,4, 5,6,7,8],
    'min_samples_leaf': [1,2,3,4,5]
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='recall')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Fit the best model to the data
best_model.fit(X_train, y_train)

# Predict the test data
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print('Best parameters:', grid_search.best_params_)
print('Accuracy Score:', accuracy)
print('Precision Score:', precision)
print('Recall Score:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', conf_matrix)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

knn = KNeighborsClassifier()

param_grid = {'n_neighbors': [1,3, 5, 7, 9], 
              'weights': ['uniform', 'distance']}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)

best_knn = grid_search.best_estimator_

y_pred = best_knn.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred)))
print('Precision Score : ' + str(precision_score(y_test, y_pred, average='weighted')))
print('Recall Score : ' + str(recall_score(y_test, y_pred, average='weighted')))
print('F1 Score : ' + str(f1_score(y_test, y_pred, average='weighted')))
print('Confusion Matrix : \n' + str(confusion_matrix(y_test, y_pred)))


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

svc = SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

qda = QuadraticDiscriminantAnalysis()

qda.fit(X_train, y_train)
pred = qda.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,pred)))
print('Precision Score : ' + str(precision_score(y_test,pred)))
print('Recall Score : ' + str(recall_score(y_test,pred)))
print('F1 Score : ' + str(f1_score(y_test,pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,pred)))

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model = QuadraticDiscriminantAnalysis()

param_grid = {
    'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print('Best parameters:', grid_search.best_params_)
print('Accuracy Score:', accuracy)
print('Precision Score:', precision)
print('Recall Score:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', conf_matrix)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

model = GradientBoostingClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


gbc = GradientBoostingClassifier()

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 7]
}

# Grid search
grid_search = GridSearchCV(gbc, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters: ", grid_search.best_params_)

# Best model
best_gbc = grid_search.best_estimator_

# Predictions
y_pred = best_gbc.predict(X_test)

# Evaluation metrics
print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred)))
print('Precision Score : ' + str(precision_score(y_test, y_pred, average='weighted')))
print('Recall Score : ' + str(recall_score(y_test, y_pred, average='weighted')))
print('F1 Score : ' + str(f1_score(y_test, y_pred, average='weighted')))
print('Confusion Matrix : \n' + str(confusion_matrix(y_test, y_pred)))


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

ridge_classifier = RidgeClassifier(alpha=1.0, solver='auto', random_state=42)
ridge_classifier.fit(X_train, y_train)
y_pred = ridge_classifier.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define the RidgeClassifier
model = RidgeClassifier()

# Define the parameter grid to search
param_grid = {
    'alpha': [0.0001,0.001,0.01,1.0,10.0,100.0]
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Fit the best model to the data
best_model.fit(X_train, y_train)

# Predict the test data
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print('Best parameters:', grid_search.best_params_)
print('Accuracy Score:', accuracy)
print('Precision Score:', precision)
print('Recall Score:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', conf_matrix)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred1 = model.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred1)))
print('Precision Score : ' + str(precision_score(y_test,y_pred1)))
print('Recall Score : ' + str(recall_score(y_test,y_pred1)))
print('F1 Score : ' + str(f1_score(y_test,y_pred1)))

print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred1)))

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

log_reg = LogisticRegression()

param_grid = {'penalty': ['l1', 'l2'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)

best_log_reg = grid_search.best_estimator_

y_pred = best_log_reg.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test, y_pred)))
print('Precision Score : ' + str(precision_score(y_test, y_pred, average='weighted')))
print('Recall Score : ' + str(recall_score(y_test, y_pred, average='weighted')))
print('F1 Score : ' + str(f1_score(y_test, y_pred, average='weighted')))
print('Confusion Matrix : \n' + str(confusion_matrix(y_test, y_pred)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

model1 = RandomForestClassifier(n_estimators=100,random_state=42)
model1.fit(X_train,y_train)
y_pred = model1.predict(X_test)

print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30], 
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print('Best parameters:', grid_search.best_params_)
print('Accuracy Score:', accuracy)
print('Precision Score:', precision)
print('Recall Score:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', conf_matrix)


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

mlp = MLPClassifier(random_state=42)

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
}

grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters: ", grid_search.best_params_)

# Best model
best_mlp = grid_search.best_estimator_

# Predictions
y_pred = best_mlp.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import numpy as np

ridge_model = Ridge(alpha=0.0001)
ridge_model.fit(X_train, y_train)
ridge_coefs = ridge_model.coef_

# Print feature names alongside coefficients
print("\nRidge Regression Coefficients:")
for feature, coef in zip(feature_names, ridge_coefs):
    print(f"{feature}: {coef}")

# Create DataFrame for plotting
data = {'Feature': feature_names, 'Coefficient': ridge_coefs}
df_plot = pd.DataFrame(data)

# Sort DataFrame by absolute coefficient values
df_plot = df_plot.reindex(df_plot['Coefficient'].abs().sort_values(ascending=False).index)

# Plotting
fig = px.bar(df_plot, x='Coefficient', y='Feature', orientation='h', 
             title='Feature Importance from Ridge Regression',
             labels={'Coefficient': 'Coefficient', 'Feature': 'Feature'})
fig.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


# Define multiple models
models = {
    "Random Forest": RandomForestClassifier(),
}

# Train each model and calculate feature importance
feature_importances = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances[name] = np.abs(model.coef_[0])
    else:
        feature_importances[name] = np.nan

# Convert feature importances to DataFrame
feature_importance_df = pd.DataFrame(feature_importances, index=X_train.columns)

# Convert to numeric values
feature_importance_df = feature_importance_df.apply(pd.to_numeric)

# Visualization using different plots
# 1. Box plot
plt.figure(figsize=(12, 8))
sns.boxplot(data=feature_importance_df)
plt.title('Feature Importance Box Plot')
plt.xticks(rotation=45)
plt.show()

# 2. Scatter plot
plt.figure(figsize=(12, 8))
for name in feature_importance_df.columns:
    plt.scatter(feature_importance_df.index, feature_importance_df[name], label=name)
plt.title('Feature Importance Scatter Plot')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# 3. Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(feature_importance_df.fillna(0), cmap='viridis', annot=True)
plt.title('Feature Importance Heatmap')
plt.show()

# # 4. Parallel coordinates plot
# from pandas.plotting import parallel_coordinates
# plt.figure(figsize=(12, 8))
# parallel_coordinates(feature_importance_df.reset_index(), 'index')
# plt.title('Feature Importance Parallel Coordinates Plot')
# plt.xticks(rotation=45)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
# plt.show()

# 5. Radar chart
from math import pi
plt.figure(figsize=(12, 8))
categories = list(feature_importance_df.index)
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
for name in feature_importance_df.columns:
    values = feature_importance_df[name].fillna(0).tolist()
    values += values[:1]
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, label=name)
    ax.fill(angles, values, 'b', alpha=0.1)
plt.title('Feature Importance Radar Chart')
plt.legend()
plt.show()

for name, model in models.items():
    model.fit(X_train, y_train)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_[0])
    else:
        feature_importance = None

    if feature_importance is not None:
        # Visualize feature importance for each model
        plt.figure(figsize=(4, 3))
        sorted_idx = np.argsort(feature_importance)[::-1]
        features = X_train.columns[sorted_idx]
        feature_importance = feature_importance[sorted_idx]
        plt.barh(features[:10], feature_importance[:10])
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title(f'Top 10 Feature Importance for {name}')
        plt.show()


import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Define multiple models
models = {
    "Random Forest": RandomForestClassifier(),
}

# Train each model and calculate feature importance
feature_importances = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances[name] = np.abs(model.coef_[0])
    else:
        feature_importances[name] = np.nan

# Convert feature importances to DataFrame
feature_importance_df = pd.DataFrame(feature_importances, index=X_train.columns)

# Convert to numeric values
feature_importance_df = feature_importance_df.apply(pd.to_numeric)

# Box plot
fig = px.box(feature_importance_df)
fig.update_traces(marker_color='lightblue')
fig.update_layout(title='Feature Importance Box Plot')
fig.show()

# Scatter plot
fig = px.scatter(feature_importance_df.reset_index().melt(id_vars=['index']), x='index', y='value', color='variable')
fig.update_traces(marker=dict(color='lightblue'))
fig.update_layout(title='Feature Importance Scatter Plot')
fig.show()

# 3. Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(feature_importance_df.fillna(0), cmap='Blues', annot=True)
plt.title('Feature Importance Heatmap')
plt.show()

# Radar chart
fig = px.line_polar(feature_importance_df.reset_index().melt(id_vars=['index']), r='value', theta='index', line_close=True)
fig.update_traces(fill='toself', line=dict(color='lightblue'))
fig.update_layout(title='Feature Importance Radar Chart')
fig.show()

for name, model in models.items():
    model.fit(X_train, y_train)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_[0])
    else:
        feature_importance = None

    if feature_importance is not None:
        # Visualize feature importance for each model
        sorted_idx = np.argsort(feature_importance)[::-1]
        features = X_train.columns[sorted_idx]
        feature_importance = feature_importance[sorted_idx]
        fig = px.bar(x=features[:10], y=feature_importance[:10], labels={'x': 'Features', 'y': 'Feature Importance'})
        fig.update_traces(marker_color='lightblue')
        fig.update_layout(title=f'Top 10 Feature Importance for {name}')
        fig.show()


import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Assuming X_train, X_test, y_train, feature_names are defined

# Train the random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Create a SHAP explainer
explainer = shap.Explainer(rf_model, X_train)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Generate a summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Generate a waterfall plot for a specific instance (change the instance_index)
instance_index = 0
shap.waterfall_plot(shap.Explanation(values=shap_values[instance_index], base_values=explainer.expected_value, data=X_test.iloc[instance_index]), max_display=30)
