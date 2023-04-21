import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def TPR_FPR(roc, clf_name='No_name', return_thresholds=False):
    """
    :param roc: array of [fpr, tpr, thresholds]
    this metric returns a dataframe where each column is the TPR and the values show FPR
    the point of this is to see how much tradeoff we have to make for each TPR
    :return: df
    """
    colmns = ['classifier'] + [i for i in range(10, 91, 10)] + [95, 100]
    df = pd.DataFrame(columns=colmns)
    df2 = pd.DataFrame(columns=colmns)

    ind_col = 0
    ans = [clf_name]
    thr = [clf_name]
    for ind, i in enumerate(roc[1]):
        i *= 100
        while i >= colmns[ind_col + 1]:
            ind_col += 1
            ans.append(roc[0][ind] * 100)
            thr.append(roc[2][ind])
            if ind_col == 11:
                break
        if ind_col == 11:
            break

    if ind_col < 11:
        for i in range(ind_col, 11):
            ans.append(110)
            thr.append(110)

    df.loc[0] = ans
    df2.loc[0] = thr
    # print(df)
    if return_thresholds:
        return df, df2
    return df

# TPR_FPR((np.arange(200), np.arange(200), np.arange(200)))
