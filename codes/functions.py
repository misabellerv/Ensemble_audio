import numpy as np
import pandas as pd

def modify_index(old_index, max_textures_per_filename, texture_selector, num_selected_textures):
    new_index = []
    
    if texture_selector == 'ALL':
        for ind in old_index:
            new_index += list(range(np.sum(max_textures_per_filename[:ind]),
                                np.sum(max_textures_per_filename[:ind+1])))
        return np.array(new_index)
    
    if texture_selector == 'LINSPACE':
        for ind in old_index:
            new_index += list(np.floor(np.linspace(np.sum(max_textures_per_filename[:ind]),
                                              np.sum(max_textures_per_filename[:ind+1]),
                                              num = num_selected_textures,
                                              endpoint = False)).astype(int))        
        return np.array(new_index)
    
    
def create_data_matrix(data_textures, index_folds, features, statistics):
    
    # data_textures (dict)
    # index_folds (dict)
    # features (list of strings)
    # statistics (list of strings) = ['mean', 'std', 'skew', 'kurt']
    
    # Pre-allocating arrays:
    
    n_rows = 0
    
    for filename in index_folds['filenames']:
        n_rows += len(data_textures[filename][features[0]][statistics[0]])
        
    n_columns = len(features)*len(statistics)
    
    X = np.zeros((n_rows,n_columns))
    
    y, filename_column, num_texture, new_features = [], [], [], []
    
    ##########
    
    cont_start = 0
    
    for ind, filename in enumerate(index_folds['filenames']):
                
        X_aux = []

        for feature in features:
            for statistic in statistics:
                X_aux.append(data_textures[filename][feature][statistic])
                
                if ind == 0:
                    new_features += [feature + '_' + statistic]
                    
        aux = len(data_textures[filename][feature][statistic])
        
        y += aux*[data_textures[filename]['label']]
        filename_column += aux*[ind]
        num_texture += range(0,aux)

        X[cont_start:cont_start+aux,:] = np.array(X_aux).T
        
        cont_start += aux

    data_matrix = pd.DataFrame(X, columns = new_features)
    data_matrix['label'] = y
    data_matrix['ind_filename'] = filename_column
    data_matrix['num_texture'] = num_texture
        
    return data_matrix