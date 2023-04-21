import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, RobustScaler, StandardScaler





data = pd.read_csv('hospital_deaths_train.csv')

def imp_nan_col(data):
    num_missing = np.array(data.columns[data.isna().any()].tolist())
    death1 = np.array(data.corr()['In-hospital_death'][num_missing])
    death = {}
    for i in range(len(num_missing)):
        death[num_missing[i]] = death1[i]
    nan_data = data[num_missing]
    nan_data = nan_data.fillna(data.median())
    corr = nan_data.corr(method='pearson')
    corr_dict = {}
    for column in corr.columns:
        corr_with_other = corr[column].sort_values(ascending=False)
        top_corr = corr_with_other.drop(column)[:3]
        corr_dict[column] = top_corr.index.tolist()
    corr_list = []
    for column in corr_dict:
        tuples = [(column, corr_dict[column][0], corr_dict[column][1]) for i in range(len(corr_dict[column])) for j in range(i+1, len(corr_dict[column]))]
        for tpl in tuples:
            if (tpl[1], tpl[0], tpl[2]) not in corr_list:
                corr_list.append(tpl)
    corr_list = list(set(corr_list))
    for i in range(len(corr_list)):
        corr_list[i] = list(corr_list[i])
    j = 0
    for i in corr_list:
        mean_corr = (corr[i[0]][i[1]] + corr[i[0]][i[2]] + corr[i[1]][i[2]]) / 3
        corr_list[j].append(mean_corr)
        j += 1
    return np.array(corr_list), death


def get_mean_var_for_drop_cols(df, cols_to_drop):
    results = []
    for col in cols_to_drop:
        mean = df[col].mean()
        var = (df[col].var()) ** (1 / 2)
        results.append({col: [mean, var]})
    return np.array(np.array(results))

def drop_data(df, target_col_name=None, degree_pol=1):
    print(df.shape)
    num_missing = df.isna().sum()
    num_missin = np.array(df.columns[df.isna().any()].tolist())
    all = df.shape[0]
    bad_col = []
    for i in range(len(num_missing)):
        if num_missing[i] / all > 0.65:
            bad_col.append(num_missin[i])

    df = df.drop(bad_col, axis=1)
    corr_list, death = imp_nan_col(df)


    for i in corr_list:
        if float(i[3]) > 0.6:
            nor = []
            for j in range(len(i) - 2):
                nor.append(death[i[j]])
            wic = np.argmax(nor)
            i = np.delete(i, wic)
            try:
                df = df.drop(i[0], axis=1)
            except:
                continue

    new_data = df

    num_missing = np.array(new_data.columns[new_data.isna().any()].tolist())
    nan_count = new_data[num_missing].isna().sum()
    num_missing_005 = num_missing[(nan_count / len(new_data["Glucose_last"]) < 0.05)]
    change_data = new_data[num_missing].reset_index(drop=True)
    change_data_005 = change_data[num_missing_005]

    for col in change_data_005:
        if col == 'Gender':
            new_data[col] = new_data[col].fillna(new_data[col].median()).astype(bool)
        else:
            new_data[col] = new_data[col].fillna(new_data[col].median())

    median_by_gender = new_data.groupby('Gender')[num_missing_005].transform('median')
    new_data[num_missing_005] = new_data[num_missing_005].fillna(median_by_gender)

    mean_var = get_mean_var_for_drop_cols(new_data, num_missing)
    for i in range(mean_var.shape[0]):
        f = np.random.normal(loc=mean_var[i][num_missing[i]][0], scale=mean_var[i][num_missing[i]][1])
        new_data[num_missing[i]] = new_data[num_missing[i]].fillna(f)

    features = df.drop(target_col_name, axis=1)
    df = new_data

    poly = PolynomialFeatures(degree=degree_pol, include_bias=True)
    poly_features = poly.fit_transform(features)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(poly_features)

    scaler = RobustScaler()
    d = scaler.fit_transform(scaled_features)

    new_data = pd.DataFrame(d, columns=[f'Feature_{i + 1}' for i in range(d.shape[1])])
    new_data[target_col_name] = df[target_col_name]
    return new_data

def scatter_mat(df):
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='hist')
    plt.show()

new_data = drop_data(data, 'In-hospital_death')

print(new_data.shape)

new_data.to_csv("new_data.csv")