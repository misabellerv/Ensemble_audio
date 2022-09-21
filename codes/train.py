### Importing Libraries ######################################################################

# General:
import pandas as pd
import numpy as np
import pickle
import sys
import yaml
import itertools

# Functions:
from functions import create_data_matrix, modify_index

# Standardization:
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer

# Feature Selection:
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif
from numpy import linalg

# Feature Reduction:
from sklearn.decomposition import PCA, FastICA, NMF

# Classifiers:
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from mrmr import mrmr_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Metrics:
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

### Parameters ##############################################################################

with open('parameters_train.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

texture_selector = params['texture_selector']['name']
num_selected_textures = params['texture_selector']['num_textures']

with open(r'D:\Users\Isabelle\Desktop\Infnet\data2_gtzan_handcrafted_textures.pk', 'rb') as f:
    data_textures = pickle.load(f)

list_features_name = list(data_textures[list(data_textures.keys())[0]].keys())[:-1]

features_set = params['features_set']['name']

if features_set == 'baseline':
    statistics = ['mean', 'std']
    features_name = [a for a in list_features_name if 'd1_' not in a and 'd2_' not in a and 'd3_' not in a and 'd4_' not in a]
elif features_set == 'sub_delta':
    statistics = ['mean', 'std']
    features_name = [a for a in list_features_name if 'd2_' not in a and 'd3_' not in a and 'd4_' not in a]
elif features_set == 'delta':
    statistics = ['mean', 'std']
    features_name = [a for a in list_features_name if 'd3_' not in a and 'd4_' not in a]
elif features_set == 'stats':
    statistics = ['mean', 'std', 'skew', 'kurt']
    features_name = [a for a in list_features_name if 'd1_' not in a and 'd2_' not in a and 'd3_' not in a and 'd4_' not in a]
elif features_set == 'sub_delta_stats':
    statistics = ['mean', 'std', 'skew', 'kurt']
    features_name = [a for a in list_features_name if 'd2_' not in a and 'd3_' not in a and 'd4_' not in a]
elif features_set == 'delta_stats':
    statistics = ['mean', 'std', 'skew', 'kurt']
    features_name = [a for a in list_features_name if 'd3_' not in a and 'd4_' not in a]

    
total_features = len(features_name) * len(statistics)

features_set = params['features_set']['name']
feature_selector = params['feature_selector']['name']
feature_reduction = params['feature_reduction']['name']
classifier = params['classifier']['name']

with open(r'D:\Users\Isabelle\Desktop\Infnet\index_folds.pkl', 'rb') as f:
    index_folds = pickle.load(f)

### Save ###################################################################################

filename_save = 'results_training'

if texture_selector == 'LINSPACE':
    filename_save = f'{filename_save}_linspace_{num_selected_textures}'
elif texture_selector == 'ALL':
    filename_save = f'{filename_save}_all'
    
filename_save = f'{filename_save}_{features_set}'

if feature_selector != '':
    filename_save = f'{filename_save}_{feature_selector}'

if feature_reduction != '':
    filename_save = f'{filename_save}_{feature_reduction}'
    
filename_save = f'{filename_save}_{classifier}'       
filename_save = f'{filename_save}.pkl'    

print(f'Results will be saved in {filename_save}')

### Creating data matrix ##################################################################

data = create_data_matrix(data_textures, index_folds, features_name, statistics)

X = data.drop(['label','ind_filename','num_texture'], axis=1)
y = data['label']


if feature_selector == 'manual':
    with open(params['feature_selector']['manual_list_filename'], 'rb') as f:
        manual_list = pickle.load(f)
    features_selected = []
    for feature in manual_list:
        features_selected.append(X.columns.get_loc(feature))
    list_percentile = [100]
    

### Encoding labels #######################################################################

filenames = list(data_textures.keys())
labels = [data_textures[filename]['label'] for filename in filenames]

encoder = LabelEncoder()
encoder.fit(labels)
y = encoder.transform(y) # y is a vector

lb = LabelBinarizer()
Y = lb.fit_transform(y) # Y is a binary matrix

### Calculating number of textures per audio track #######################################

max_textures_per_filename = np.array(data[['ind_filename','num_texture']].groupby(['ind_filename']).max()['num_texture']) + 1

### Training #############################################################################

results_training = {}

for ind_set in index_folds['sets'].keys():
    
    accuracy_dict = {}
    
    print(f'Model: {ind_set}/100\n')
    
    ind_train = modify_index(index_folds['sets'][ind_set]['train'], max_textures_per_filename, texture_selector, num_selected_textures)  

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[ind_train])
    
    y_train = y[ind_train]
    
    Y_train = Y[ind_train,:]
    
    ### Feature Selector:
    
    # Parameters:
    
    if feature_selector == '':
        fs_params = {'fs':'N'}
    elif feature_selector == 'anova':
        fs_params = {'fs':['anova']}
    elif feature_selector == 'mutual':
        fs_params = {'fs':['mutual']}
    elif feature_selector == 'somp':
        fs_params = {'fs':['somp'],
                     'somp_n': params['feature_selector']['somp_n'],
                     'somp_K': params['feature_selector']['somp_K']}
    elif feature_selector == 'mrmr':
        fs_params = {'fs':['mrmr'],
                    'mrmr_K': params['feature_selector']['mrmr_K']}
    elif feature_selector == 'manual':
        fs_params = {'fs':['manual']}
        
    keys, values = zip(*fs_params.items())
    perm_fs_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
    # Function:
    
    dict_string_params = {}  # Initialize string_params !!!
    
    for fs_param in perm_fs_params:
        
        dict_string_params.update(fs_param)
        
        if feature_selector == '':
            features_selected = list(range(0,total_features))
            list_percentile = [100]
        elif feature_selector == 'anova':
            fs = f_classif(X_train, y_train)
            aux = sorted([(x,ind) for (ind,x) in enumerate(fs[0])], reverse=True)
            features_selected = [ind for x, ind in aux]
            list_percentile = params['feature_selector']['percentile']
        elif feature_selector == 'mutual':
            fs = mutual_info_classif(X_train, y_train)
            aux = sorted([(x,ind) for (ind,x) in enumerate(fs)], reverse=True)
            features_selected = [ind for x, ind in aux]
            list_percentile = params['feature_selector']['percentile']
        elif feature_selector == 'somp':
            K = fs_param['somp_K'] #total_features
            n_dict = {}
            n = fs_param['somp_n']
            R = Y_train
            uIdx = list(range(0,total_features))
            sIdx = []
            X_ = X_train.copy()
            Y_ = Y_train.copy()
            for k in range(0, K):
                idx = np.argmax([linalg.norm(np.array([x_j]) @ R) / linalg.norm(np.array([x_j]).T) for x_j in X_[:,uIdx].T])
                sIdx.append(uIdx[idx])
                uIdx.remove(uIdx[idx])
                A = np.linalg.inv(X_[:,sIdx].T @ X_[:,sIdx]) @ X_[:,sIdx].T @ Y_
                R = Y_ - n * X_[:,sIdx] @ A
            features_selected = sIdx
            list_percentile = [100] #params['feature_selector']['percentile']
        elif feature_selector == 'mrmr':
            features_selected = mrmr_classif(X_train, y_train, K = fs_param['mrmr_K']) #total_features)
            list_percentile = [100] #params['feature_selector']['percentile']
            
            
        for p in list_percentile:
            dict_string_params.update({'percentile':p})
            aux_p = int(total_features * p / 100)           
            X_train_fs = X_train[:,features_selected[:aux_p]]
            
        
            ### Feature Reduction:
        
            # Parameters:
    
            if feature_reduction == '':
                fr_params = {'fr':'N'}
            elif feature_reduction == 'pca':
                fr_params = {'fr':['pca'],
                             'n_components': params['feature_reduction']['n_components']}
            elif feature_reduction == 'fastica':
                fr_params = {'fr':['fastica'],
                             'n_components': params['feature_reduction']['n_components']}
            
            keys, values = zip(*fr_params.items())
            perm_fr_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
            # Function:
    
            for fr_param in perm_fr_params:
            
                dict_string_params.update(fr_param)
            
                if feature_reduction == '':
                    X_train_fr = X_train_fs.copy()
                elif feature_reduction == 'pca':
                    fr = PCA(n_components=fr_param['n_components'])
                    X_train_fr = fr.fit_transform(X_train_fs)
                elif feature_reduction == 'fastica':
                    fr = FastICA(n_components=fr_param['n_components'],
                                 max_iter=params['fastica_parameters']['max_iter'])
                    X_train_fr = fr.fit_transform(X_train_fs)
                    
                    
                ### Classifier:
        
                # Parameters:
            
                if classifier == 'logistic':
                    clf_params = params['logistic_parameters']
                elif classifier == 'knn':
                    clf_params = params['knn_parameters']
                elif classifier == 'svm':
                    clf_params = params['svm_parameters']
                elif classifier == 'rf':
                    clf_params = params['rf_parameters']    
                elif classifier == 'lgboost':
                    clf_params = params['lgboost_parameters']
                elif classifier == 'gnb':
                    clf_params = params['gnb_parameters']
                elif classifier == 'lda':
                    clf_params = params['lda_parameters']
                elif classifier == 'qda':
                    clf_params = params['qda_parameters']
                elif classifier == 'mlp':
                    clf_params = params['mlp_parameters']
                elif classifier == 'svm-linear':
                    clf_params = params['svm-linear_parameters']
                elig classifier == 'stacking'
                    clf_params = params['stacking_parameters']
                    
                keys, values = zip(*clf_params.items())
                perm_clf_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
                    
                # Function:
    
                for clf_param in perm_clf_params:
            
                    dict_string_params.update(clf_param)
            
                    if classifier == 'logistic':
                        clf = LogisticRegression(max_iter=clf_param['max_iter'], C=clf_param['C'])
                    elif classifier == 'knn':
                        clf = KNeighborsClassifier(n_neighbors=clf_param['n_neighbors'])
                    elif classifier == 'svm':
                        clf = SVC(kernel=clf_param['kernel'], gamma=clf_param['gamma'], C=clf_param['C'])
                    elif classifier == 'rf':
                        clf = RandomForestClassifier(n_estimators=clf_param['n_estimators'])
                    elif classifier == 'lgboost':
                        clf = LGBMClassifier(n_estimators=clf_param['n_estimators'], learning_rate=clf_param['learning_rate'])
                    elif classifier == 'gnb':
                        clf = GaussianNB(var_smoothing=clf_param['var_smoothing'])
                    elif classifier == 'lda':
                        clf = LinearDiscriminantAnalysis(solver=clf_param['solver'], shrinkage=clf_param['shrinkage'])
                    elif classifier == 'qda':
                        clf = QuadraticDiscriminantAnalysis(reg_param=clf_param['reg_param'])
                    elif classifier == 'mlp':
                        clf = MLPClassifier(hidden_layer_sizes=tuple(clf_param['hidden_layer_sizes']),
                                            random_state=1, max_iter=clf_param['max_iter'], activation=clf_param['activation'])
                    elif classifier == 'svm-linear':
                        clf = LinearSVC(max_iter=clf_param['max_iter'], C=clf_param['C'], dual=False)
                        
                    clf.fit(X_train_fr, y_train)
                    
                    # Predicting per Song:
                    
                    y_pred_total = [] 
                    y_real_total = []
                    
                    for ind_song in index_folds['sets'][ind_set]['val']:
                        ind_val_textures_per_song = modify_index([ind_song], max_textures_per_filename, 'ALL', num_selected_textures)
                        
                        X_val_textures_per_song = X.iloc[ind_val_textures_per_song]
                        y_val_textures_per_song = y[ind_val_textures_per_song]
                        
                        X_val_textures_per_song = scaler.transform(X_val_textures_per_song)
                        X_val_textures_per_song_fs = X_val_textures_per_song[:,features_selected[:aux_p]]
                        
                        if feature_reduction == '':
                            X_val_textures_per_song_fr = X_val_textures_per_song_fs.copy()
                        else:
                            X_val_textures_per_song_fr = fr.transform(X_val_textures_per_song_fs)   
                                                    
                        y_pred_textures_per_song = clf.predict(X_val_textures_per_song_fr)
                    
                        y_pred_total += [stats.mode(y_pred_textures_per_song)[0]]
                        y_real_total += [stats.mode(y_val_textures_per_song)[0]]
                        
                    accuracy_dict[str(dict_string_params)] = accuracy_score(y_real_total, y_pred_total)
                    
                    print(dict_string_params)
 
    results_training[ind_set] = accuracy_dict

    
with open(filename_save, 'wb') as f:
    pickle.dump(results_training, f)