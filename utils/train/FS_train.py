import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC

# clf = joblib.load(r'D:\workplace\test\shape\ex\model\face_shape_classifier.pkl')
home_path = r'D:\workplace\ex'

heart_df = pd.read_pickle(home_path + '\\heart_.pkl')
oval_df = pd.read_pickle(home_path + '\\oval_.pkl')
round_df = pd.read_pickle(home_path + '\\round_.pkl')
square_df = pd.read_pickle(home_path + '\\square_.pkl')
dia_df = pd.read_pickle(home_path + '\\dia_.pkl')
long_df = pd.read_pickle(home_path + '\\long_.pkl')

colum_names = [
               'curvature_radius',    # 곡률 반경
               'chin_angle_inclease', # 턱 선 각도 증가률
               'chin_gradient',       # 턱 선 기울기 감소율 차이
               'chin_born_angle',     # 턱 골격 각도
               'chin_shape_angle',    # 턱 모양 각도
               'brow_shape_angle',    # 이마 모양 각도
               'ratio_width_height',  # 얼굴 너비 대비 얼굴 길이
               'raito_brow_chin',     # 얼굴 이마 길이 대비 턱 사이 길이
               'ratio_width_chin'     # 얼굴 너비 대비 턱 사이 길이
               ]


print(len(heart_df)) # 1320
print(len(oval_df)) # 1014
print(len(round_df)) # 468
print(len(square_df)) # 468
print(len(dia_df)) # 468
print(len(long_df)) # 468

# heart_df = heart_df.loc[:449]
# oval_df = oval_df.loc[:449]
# round_df = round_df.loc[:389]
# square_df = square_df.loc[:389]

min_max = []
def min_max_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(normalized_num)
    return normalized, np.mean(lst), np.std(lst)

df = pd.concat([heart_df,oval_df,round_df,square_df,dia_df,long_df], ignore_index=True)
for col in colum_names:
    col_list = list(df[col])
    normalized, Min, Max = min_max_normalize(col_list)
    min_max.append((col, Min, Max))
    df[col] = normalized
print(min_max)
labels = ['heart', 'oval', 'round', 'square', 'dia', 'long']
x_data = df[['curvature_radius', 'chin_angle_inclease', 'chin_gradient', 'chin_born_angle', 'chin_shape_angle',
               'brow_shape_angle', 'ratio_width_height', 'raito_brow_chin', 'ratio_width_chin']]
y_data = df['face shape']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.2,
                                                    # random_state=777,
                                                    # stratify=y_data
                                                    )
xgb_model1 = xgb.XGBClassifier(n_estimators=100)
# 후보 파라미터 선정
params = {'max_depth':[3,5,7], 'min_child_weight':[1,3], 'colsample_bytree':[0.5,0.75]}
# gridsearchcv 객체 정보 입력(어떤 모델, 파라미터 후보, 교차검증 몇 번)
gridcv = GridSearchCV(xgb_model1, param_grid=params, cv=3)
# 파라미터 튜닝 시작
gridcv.fit(x_train, y_train, early_stopping_rounds=30, eval_metric='auc', eval_set=[(x_test, y_test)])
#튜닝된 파라미터 출력
print(gridcv.best_params_)
# 1차적으로 튜닝된 파라미터를 가지고 객체 생성
xgb_model = xgb.XGBClassifier(objective='multi:softmax',
                              num_class=6,
                              n_estimators=2000,
                              learning_rate=0.02,
                              max_depth=3,
                              min_child_weight=1,
                              colsample_bytree=0.5
                              ).fit(x_train, y_train,
                                    eval_metric='mlogloss',
                                    early_stopping_rounds=100,
                                    eval_set = [(x_test, y_test)],
                                    verbose =1
                                    )
y_score = xgb_model.predict_proba(x_test)
# 특성 중요도 시각화
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)
# vis = ROCAUC(xgb_model, classes=[0, 1, 2, 3, 4, 5], micro=True, macro=True, per_class=True)
# vis.fit(x_train, y_train)
# vis.score(x_train, y_train)
# vis.show()

y_test = y_test.reset_index(drop=True)

# ROC & AUC
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# n_samples = 6
# for i in range(n_samples):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# # Plot of a ROC curve for a specific class
# plt.figure(figsize=(15, 5))
# for idx, i in enumerate(range(n_samples)):
#     plt.subplot(131+idx)
#     plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Class %0.0f' % idx)
#     plt.legend(loc="lower right")
# plt.show()

print(xgb_model.score(x_train, y_train)) #
print(xgb_model.score(x_test, y_test)) #0.75
# GB = GradientBoostingClassifier(n_estimators=1000,
#                                 criterion='friedman_mse',
#                                 max_depth=3,
#                                 max_leaf_nodes=None,
#                                 min_samples_split=2,
#                                 min_samples_leaf=1,
#                                 learning_rate=0.1,
#                                 max_features=None
#                                 ).fit(x_train, y_train)
# print(GB.score(x_train, y_train)) #
# print(GB.score(x_test, y_test)) #0.75
joblib.dump(xgb_model, home_path + '\\model\\face_shape_classifier_XGB.pkl')