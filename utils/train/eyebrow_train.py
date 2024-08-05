import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
# clf = joblib.load(r'D:\workplace\test\shape\ex\model\face_shape_classifier.pkl')
home_path = r'D:\workplace\test\shape\ex\train2'

arch_df = pd.read_pickle(home_path + '\\arch_.pkl')
flat_df = pd.read_pickle(home_path + '\\flat_.pkl')
up_df = pd.read_pickle(home_path + '\\up_.pkl')

# colum_names = ['distance', 'ef0', 'ef1', 'ef2', 'ef3', 'ef4', 'rad0', 'rad1', 'rad2', 'rad3',
#                'rad_ratio0', 'rad_ratio1', 'rad_ratio2']
colum_names = ['ef0', 'ef1', 'ef2', 'ef3', 'ef4', 'rad0', 'rad1', 'rad2', 'rad3',
               'rad_ratio0', 'rad_ratio1', 'rad_ratio2']

# arch_df = arch_df.loc[:499]
# deep_arch_df = deep_arch_df.loc[:499]
# flat_df = flat_df.loc[:499]
# up_df = up_df.loc[:499]

min_max = []
def min_max_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(normalized_num)
    return normalized, np.mean(lst), np.std(lst)

df = pd.concat([arch_df,flat_df,up_df], ignore_index=True)
for col in colum_names:
    col_list = list(df[col])
    normalized, Min, Max = min_max_normalize(col_list)
    min_max.append((col, Min, Max))
    df[col] = normalized
print(min_max)
labels = ['arch', 'flat', 'up']
x_data = df[['ef0', 'ef1', 'ef2', 'ef3', 'ef4', 'rad0', 'rad1', 'rad2', 'rad3',
               'rad_ratio0', 'rad_ratio1', 'rad_ratio2']]
y_data = df['eyebrow_shape']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.2,
                                                    random_state=777,
                                                    stratify=y_data)
xgb_model = xgb.XGBClassifier(objective='multi:softmax',
                              num_class=3,
                              n_estimators=500,
                              learning_rate=0.1,
                              max_depth=3,
                              ).fit(x_train, y_train,
                                    eval_metric='mlogloss',
                                    eval_set = [(x_test, y_test)],
                                    verbose =1
                                    )
print(xgb_model.score(x_train, y_train)) #
print(xgb_model.score(x_test, y_test)) # 0.75

joblib.dump(xgb_model, home_path + '\\model\\eyebrow_shape_3_classifier_XGB.pkl')