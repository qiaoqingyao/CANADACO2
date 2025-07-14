import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import KernelKMeans,TimeSeriesKMeans,silhouette_score,KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance,TimeSeriesResampler
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error
from xgboost import XGBRegressor
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import cross_val_score,LeaveOneOut,GridSearchCV, KFold,TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFECV
import shap

pd.set_option('display.float_format', lambda x: '%.3f' % x)
plt.rcParams.update({'font.size': 10})


df = pd.read_excel(r'data\USA_Transport-Energy Consum and CO2 (2024-10-09).xls')
df =df.set_index('Year')
df = df.fillna(df.mean())
df.columns =[x.split(' ')[0] for x in df.columns]
df= df.drop(2020,axis=0)
# Statistical description
Des = df.describe().T.round(2)
# Plot
y = df['TCT']
x = df.drop('TCT',axis=1)
x_ = x.iloc[:-1,:]
y_ = y.iloc[1:]
# plt data
fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    ax.plot(df.iloc[:,i])
    ax.set_title(df.columns[i])
plt.tight_layout()
plt.show()

'''
hierarchical clustering
Using TimeSeriesKmeans to calculate silhouette scores for different clusters numbers to 
determines the best cluster number setting.
'''

x_ts = to_time_series_dataset(x.T)
x_ts_scaled = TimeSeriesScalerMeanVariance().fit_transform(x_ts)

cluster_range = range(2, 15)
# Fit models and calculate silhouette scores for each number of clusters
silhouette_scores = []
for n_clusters in cluster_range:
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=8, n_init=2,random_state=2025)
    y_pred = model.fit_predict(x_ts_scaled)
    silhouette = silhouette_score(x_ts_scaled, y_pred)
    silhouette_scores.append(silhouette)
# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()
# training and clustering
model = TimeSeriesKMeans(n_clusters=8, metric="dtw", max_iter=8, n_init=2,random_state=2025)
y_pred = model.fit_predict(x_ts_scaled)
# Visualizing the clustering results
plt.figure(figsize=(12, 16))
for yi in range(8):
    plt.subplot(n_clusters, 1, 1 + yi)
    for xx in x_ts_scaled[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(model.cluster_centers_[yi].ravel(), "r-")
    plt.title("Cluster %d" % (yi + 1))
plt.tight_layout()
plt.show()
clustering = pd.DataFrame({'variable': x.columns,
                           'cluster': y_pred})
#plotting hierarchical feature clustering
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled,columns = x.columns)
x_tranpose = x_scaled.transpose()

linked = linkage(x_tranpose, method='ward', metric='euclidean')
df_linked = pd.DataFrame(linked, columns = ['c1','c2','distance','size'])
df_linked [['c1','c2','size']]= df_linked[['c1','c2','size']].astype(int)

plt.figure(figsize=(10, 6))
dendrogram(linked,
           orientation='top',
           labels=x_tranpose.index,
           distance_sort='descending',
           show_leaf_counts=True)

plt.title("Hierarchical feature clustering")
plt.xlabel("features")
plt.ylabel("Ward's distance")
plt.show()

'''
 burota feature selection to determine which information is important
'''
ranking = []
for i in range(3):
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=i)
    boru = BorutaPy(rf,verbose=2)
    sel = boru.fit_transform(x_.values,y_.values)
    ranking.append(boru.ranking_)
    print('========round ' +str(i) +"========")
boruta_feature = pd.DataFrame({
    'variable': x_.columns,
    'ranking': ranking[0]})
boruta_feature =boruta_feature[boruta_feature['ranking']==1]

# correlation
for i in x_:
    corr_coef, p_value = pearsonr(x_[i].values, y_['TCT'])
    print("the correlation between %s and CO2 is %0.5f and the p-vale is %0.5f" % (i, corr_coef, p_value))

for i in x_:
    corr_coef, p_value = spearmanr(x_[i].values, y_)
    print(p_value)

'''
Feature selection based on boruta, correlation analysis, clustering was conducted outside the python script.
Machine learning based on the selected feature

'''

# data preprocess
x_selected = df[['TPEI','FFT',"AVMT","APDI",'UR','TCT']]
columns_name = x_selected.columns
scaler = StandardScaler()
selected= scaler.fit_transform(x_selected)
selected = pd.DataFrame(selected,columns = columns_name)

x__ = scaler.fit_transform(df)
x__ = pd.DataFrame(x__,columns = df.columns)

def shift(df, step):
    for i in np.arange(1, step):
        df_ = df
        col_name = 'Lag_{}_TCT'.format(i)
        df_[col_name] = df_['TCT'].shift(i)
        df_.interpolate(method = 'linear', limit_direction = 'backward',axis=0, limit = 2, inplace = True)
    return df_

x_selected_ = shift(selected, 3)
x_selected_ = x_selected_.iloc[:-1,:]
x__ = shift(x__,3)
x__ =x__ .iloc[:-1,:]

x_selected_.columns = ['TPEI', 'FFT', 'AVMT', 'APDI', 'UR', 'TCT(t)', 'TCT(t-1)', 'TCT(t-2)']
# our proposed feature selection
x_train_1, x_test_1, y_train, y_test = train_test_split(x_selected_, y_, test_size=0.2 ,random_state=10, shuffle=False)

# original dataset
x_train_2, x_test_2, y_train, y_test = train_test_split(x__, y_, test_size=0.2 ,random_state=10, shuffle=False)
# GV grid_search
# recursive feature elimination
rfe = RFECV(RandomForestRegressor(),cv=10,scoring="neg_mean_squared_error")
rfe.fit(x_train_2,y_train)
rfe_feature = np.array(x_train_2.columns)[rfe.get_support()]
x_train_rf = x_train_2[rfe_feature]
x_test_rf = x_test_2[rfe_feature]

model_Xgboost = XGBRegressor()
param_grid_Xgboost = {
    'n_estimators': [10, 30, 50, 100, 200],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    'max_depth': [1, 2, 3,4,5]}

model_Svr = SVR()
param_grid_Svr = {'C': [1, 10, 12, 14, 16, 18, 20, 22],
            'gamma': [0.001, 0.01, 0.1, 1, 2, 5],
            'epsilon': [0.001, 0.01, 0.1, 1, 2, 4],
            'kernel': ("rbf", "poly", "sigmoid")}

model_ElasticNet = ElasticNet()
param_grid_ElasticNet = {'alpha'     : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'l1_ratio'  :  np.arange(0.0,1.00,0.10),
                'tol'       : [0.0001,0.001,0.01]}

model_Ann = MLPRegressor()
param_grid_Ann = {'hidden_layer_sizes': [(5,),(10,),(50,),(80,),(100,)],
          'activation': ['relu','tanh','logistic'],
          'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
          'learning_rate': [ 'adaptive', 'invscaling', 'constant'],
          'solver': ['adam']}

def GSCV(model, param_grid,x_train):
    tscv = TimeSeriesSplit(n_splits=10)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    # Fit the grid search object on the training data
    grid_search.fit(x_train, y_train)
    result = pd.DataFrame(grid_search.cv_results_)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params, result

# our data performance
best_model_Xgboost,best_params_Xgboost,results_Xgboost= GSCV(model_Xgboost, param_grid_Xgboost,x_train_1)
best_model_Svr,best_params_Svr,results_Svr= GSCV(model_Svr, param_grid_Svr,x_train_1)
best_model_ElasticNet,best_params_ElasticNet,results_ElasticNet= GSCV(model_ElasticNet, param_grid_ElasticNet,x_train_1)
best_model_Ann,best_params_Ann,results_Ann= GSCV(model_Ann, param_grid_Ann,x_train_1)

# original data performance

best_model_Xgboost2,best_params_Xgboost2,results_Xgboost2= GSCV(model_Xgboost, param_grid_Xgboost,x_train_2)
best_model_Svr2,best_params_Svr2,results_Svr2= GSCV(model_Svr, param_grid_Svr,x_train_2)
best_model_ElasticNet2,best_params_ElasticNet2,results_ElasticNet2= GSCV(model_ElasticNet, param_grid_ElasticNet,x_train_2)
best_model_Ann2,best_params_Ann2,results_Ann2= GSCV(model_Ann, param_grid_Ann,x_train_2)

# recursive data performance

best_model_Xgboost3,best_params_Xgboost3,results_Xgboost3= GSCV(model_Xgboost, param_grid_Xgboost,x_train_rf)
best_model_Svr3,best_params_Svr3,results_Svr3= GSCV(model_Svr, param_grid_Svr,x_train_rf)
best_model_ElasticNet3,best_params_ElasticNet3,results_ElasticNet3= GSCV(model_ElasticNet, param_grid_ElasticNet,x_train_rf)
best_model_Ann3,best_params_Ann3,results_Ann3= GSCV(model_Ann, param_grid_Ann,x_train_rf)

def training(best_model,x_train):
    y_pred = best_model.predict(x_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)
    mape = mean_absolute_percentage_error(y_train, y_pred)
    error = y_pred - y_train
    std = np.std(error)
    return std,rmse, mae, mape,y_pred
def testing(best_model,x_test):
    y_pred = best_model.predict(x_test)
    error = y_pred - y_test
    std = np.std(error)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return std,rmse,mae,mape,y_pred




training(best_model_Xgboost,x_train_1)
training(best_model_Ann,x_train_1)
training(best_model_Svr,x_train_1)
training(best_model_ElasticNet,x_train_1)
training(best_model_Xgboost2,x_train_2)
training(best_model_Ann2,x_train_2)
training(best_model_Svr2,x_train_2)
training(best_model_ElasticNet2,x_train_2)
training(best_model_Xgboost3,x_train_rf)
training(best_model_Ann3,x_train_rf)
training(best_model_Svr3,x_train_rf)
training(best_model_ElasticNet3,x_train_rf)


testing(best_model_Xgboost,x_test_1)
testing(best_model_Ann,x_test_1)
testing(best_model_Svr,x_test_1)
testing(best_model_ElasticNet,x_test_1)
testing(best_model_Xgboost2,x_test_2)
testing(best_model_Ann2,x_test_2)
testing(best_model_Svr2,x_test_2)
testing(best_model_ElasticNet2,x_test_2)
testing(best_model_Xgboost3,x_test_rf)
testing(best_model_Ann3,x_test_rf)
testing(best_model_Svr3,x_test_rf)
testing(best_model_ElasticNet3,x_test_rf)

# shap analysis

shap_model = best_model_ElasticNet.fit(x_selected_, y_,)
explainer = shap.explainers.Linear(shap_model, x_selected_)
shap_values = explainer(x_selected_)
fig, ax = plt.subplots()
shap.plots.beeswarm(shap_values)
fig.tight_layout()
fig.show()

fig, ax = plt.subplots()
shap.plots.heatmap(shap_values)
fig.tight_layout()
fig.show()

fig, ax = plt.subplots(figsize=(16,10))
shap.plots.scatter(shap_values, ylabel='SHAP Value of TCT(t+1)')


