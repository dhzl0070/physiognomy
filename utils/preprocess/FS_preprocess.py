import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

home_path = r'D:\workplace\ex'

heart_df = pd.read_pickle(home_path + '\\heart.pkl')
oval_df = pd.read_pickle(home_path + '\\oval.pkl')
round_df = pd.read_pickle(home_path + '\\round.pkl')
square_df = pd.read_pickle(home_path + '\\square.pkl')
dia_df = pd.read_pickle(home_path + '\\dia.pkl')
long_df = pd.read_pickle(home_path + '\\long.pkl')

colum_names = [
               'curvature_radius',    # 곡률 반경
               'chin_angle_inclease', # 턱 선 각도 증가률
               'chin_gradient',       # 턱 선 기울기 감소율 차이
               'chin_born_angle',     # 턱 골격 각도
               'chin_shape_angle',    # 턱 모양 각도
               'brow_shape_angle',    # 이마 모양 각도
               'ratio_width_height',  # 얼굴 너비 대비 얼굴 길이
               'raito_brow_chin',     # 얼굴 이마 길이 대비 턱 사이 길이
               'ratio_width_brow'     # 얼굴 너비 대비 얼굴 이마 길이
               ]

print(len(heart_df),
len(oval_df), # 978
len(round_df), # 418
len(square_df), # 467
len(dia_df),
len(long_df))

# 곡률 반경
heart_df['curvature_radius'].describe()
oval_df['curvature_radius'].describe()
round_df['curvature_radius'].describe()
square_df['curvature_radius'].describe()
dia_df['curvature_radius'].describe()
long_df['curvature_radius'].describe()

heart_df['curvature_radius'].values.tolist()
heart_df.drop(heart_df[heart_df['curvature_radius'] > 300].index, inplace=True)
heart_df.drop(heart_df[heart_df['curvature_radius'] < 100].index, inplace=True)
heart_df.reset_index(inplace=True)
oval_df.drop(oval_df[oval_df['curvature_radius'] > 300].index, inplace=True)
oval_df.drop(oval_df[oval_df['curvature_radius'] < 100].index, inplace=True)
oval_df.reset_index(inplace=True)
round_df.drop(round_df[round_df['curvature_radius'] < 150].index, inplace=True)
round_df.reset_index(inplace=True)

square_df.drop(square_df[square_df['curvature_radius'] >= 250].index, inplace=True)
square_df.reset_index(inplace=True)
dia_df.drop(dia_df[dia_df['curvature_radius'] >= 300].index, inplace=True)
dia_df.reset_index(inplace=True)
long_df.drop(long_df[long_df['curvature_radius'] >= 300].index, inplace=True)
long_df.drop(long_df[long_df['curvature_radius'] < 150].index, inplace=True)
long_df.reset_index(inplace=True)

# 턱 선 각도 증가률
heart_df['chin_angle_inclease'].describe()
oval_df['chin_angle_inclease'].describe()
round_df['chin_angle_inclease'].describe()
square_df['chin_angle_inclease'].describe()
dia_df['chin_angle_inclease'].describe()
long_df['chin_angle_inclease'].describe()
# 턱 선 기울기 감소율 차이
heart_df['chin_gradient'].describe()
oval_df['chin_gradient'].describe()
round_df['chin_gradient'].describe()
square_df['chin_gradient'].describe()
dia_df['chin_gradient'].describe()
long_df['chin_gradient'].describe()
# 턱 골격 각도
heart_df['chin_born_angle'].describe()
oval_df['chin_born_angle'].describe()
round_df['chin_born_angle'].describe()
square_df['chin_born_angle'].describe()
dia_df['chin_born_angle'].describe()
long_df['chin_born_angle'].describe()
# 턱 모양 각도
heart_df['chin_shape_angle'].describe()
oval_df['chin_shape_angle'].describe()
round_df['chin_shape_angle'].describe()
square_df['chin_shape_angle'].describe()
dia_df['chin_shape_angle'].describe()
long_df['chin_shape_angle'].describe()
# 이마 모양 각도
heart_df['brow_shape_angle'].describe()
oval_df['brow_shape_angle'].describe()
round_df['brow_shape_angle'].describe()
square_df['brow_shape_angle'].describe()
dia_df['brow_shape_angle'].describe()
long_df['brow_shape_angle'].describe()
# 얼굴 너비 대비 얼굴 길이
heart_df['ratio_width_height'].describe()
oval_df['ratio_width_height'].describe()
round_df['ratio_width_height'].describe()
square_df['ratio_width_height'].describe()
dia_df['ratio_width_height'].describe()
long_df['ratio_width_height'].describe()

# heart_df = heart_df.loc[:449]
# oval_df = oval_df.loc[:449]
# round_df = round_df.loc[:449]
# square_df = square_df.loc[:449]

# joblib.dump(clf, home_path + '\\model\\face_shape_classifier.pkl')
heart_df.to_pickle(home_path + '\\heart_.pkl')
oval_df.to_pickle(home_path + '\\oval_.pkl')
round_df.to_pickle(home_path + '\\round_.pkl')
square_df.to_pickle(home_path + '\\square_.pkl')
dia_df.to_pickle(home_path + '\\dia_.pkl')
long_df.to_pickle(home_path + '\\long_.pkl')