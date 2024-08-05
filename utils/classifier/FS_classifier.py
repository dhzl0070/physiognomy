from face_shape_classify.main_func import *
from face_shape_classify.main_func import FaceShape as FF
import shutil
import glob, os
from tqdm import tqdm
import pandas as pd
import joblib
import warnings
warnings.filterwarnings(action='ignore')

model = joblib.load(r'D:\workplace\test\shape\ex\train2\model\face_shape_classifier_XGB_final.pkl')
img_dir = rf'D:\data\yolo data\9label\images\HF/female'
home_path = r'D:\workplace\test\shape\valid\faceshape'
heart_path = home_path + r'\heart'
oval_path = home_path + r'\oval'
round_path = home_path + r'\round'
dia_path = home_path + r'\dia'
long_path = home_path + r'\long'
square_path = home_path + r'\square'
none_path = home_path + r'\none'
ex_path = home_path + r'\ex'

labels = ['heart', 'oval', 'round', 'square', 'dia', 'long']
h_cnt = 0
o_cnt = 0
r_cnt = 0
s_cnt = 0
d_cnt = 0
l_cnt = 0
n_cnt = 0
for f in tqdm(glob.glob(os.path.join(img_dir, "*.*"))):
    try:
        # 이미지 파일의 경우을 사용하세요.:
        img = [f]
        img_name = img[0].split(".")[0].split("\\")[-1]

        rotate_img = FF.iris_rotate(img)
        img_trim = FF.landmark(rotate_img)
        annotated_image, landmark_tuple = FF.feature_extract(img_trim)

        left_face_direction = int(landmark_tuple[23][0] - landmark_tuple[13][0])
        right_face_direction = int(landmark_tuple[13][0] - landmark_tuple[22][0])
        face_direction = abs(right_face_direction - left_face_direction)
        if face_direction <= 150:
            a5 = round(angle3(landmark_tuple, 18, -1, 19))  # 턱의 모양 각도
            a6 = round(angle3(landmark_tuple, 0, 4, 1))  # 이마 사이각
            a7 = round(angle3(landmark_tuple, 4, 0, 5))  # 얼굴 모양 각도
            a4_R = round(angle3(landmark_tuple, 22, 18, 8))  # 턱 골격 각도
            a4_L = round(angle3(landmark_tuple, 23, 19, 9))  # 턱 골격 각도

            # 얼굴 길이
            A1 = distance(landmark_tuple, 0, 1)  # 얼굴 가로 길이
            A2 = distance(landmark_tuple, 2, 3)  # 얼굴 이마 길이
            A3 = distance(landmark_tuple, 4, 5)  # 얼굴 세로 길이
            A4 = distance(landmark_tuple, 6, 7)  # 얼굴 턱 사이 길이
            A6 = distance(landmark_tuple, 10, 11)  # 얼굴 턱 크기(길이)
            A7 = distance(landmark_tuple, 4, 12)  # 이마끝부터 눈썹까지 세로길이
            A8 = distance(landmark_tuple, 12, 13)  # 눈썹끝부터 코밑까지 세로길이
            A9 = distance(landmark_tuple, 13, 5)  # 코밑부터 턱끝까지 세로길이
            # 얼굴 비율
            # R1 = f'{(round((A7 / A3), 2))} : {(round((A8 / A3), 2))} : {(round((A9 / A3), 2))}'  # 얼굴 길이 대비 3등분 비율
            R2 = (round((A3 / A1), 2))  # 얼굴 너비 대비 얼굴 길이
            R5 = (round((A2 / A1), 2))  # 얼굴 너비 대비 얼굴 이마 길이
            R6 = (round((A4 / A1), 2))  # 얼굴 너비 대비 턱 사이 길이

            if a4_R > a4_L:
                a4 = a4_L
                lin2 = (round(lin(landmark_tuple, 9, 21), 2))
                lin3 = (round(lin(landmark_tuple, 21, 19), 2))
                lin4 = (round(lin(landmark_tuple, 19, 7), 2))
                lin5 = (round(lin(landmark_tuple, 7, 23), 2))
                linM1 = round((lin5 - lin4) / lin4 * 100)
                lin_total = round((lin4 - lin3) / lin3 * 100)

                a0 = round(angle3(landmark_tuple, 13, 5, 23))  # 132
                a1 = round(angle3(landmark_tuple, 13, 5, 7))  # 58
                a2 = round(angle3(landmark_tuple, 13, 5, 19))  # 172
                a3 = round(angle3(landmark_tuple, 13, 5, 21))  # 176
                chin_total = round((a3 - a0) / a0 * 100)
                r_total = r_interesection(landmark_tuple, 7, 19, 21)
            else:
                a4 = a4_R
                lin2 = (round(lin(landmark_tuple, 8, 20), 2))
                lin3 = (round(lin(landmark_tuple, 20, 18), 2))
                lin4 = (round(lin(landmark_tuple, 18, 6), 2))
                lin5 = (round(lin(landmark_tuple, 6, 22), 2))
                linM1 = round((lin5 - lin4) / lin4 * 100)
                lin_total = round((lin4 - lin3) / lin3 * 100)

                a0 = round(angle3(landmark_tuple, 13, 5, 22))  # 132
                a1 = round(angle3(landmark_tuple, 13, 5, 6))  # 58
                a2 = round(angle3(landmark_tuple, 13, 5, 18))  # 172
                a3 = round(angle3(landmark_tuple, 13, 5, 20))  # 176
                chin_total = round((a3 - a0) / a0 * 100)
                r_total = r_interesection(landmark_tuple, 6, 18, 20)

            if R2 <= 1.3:
                if (lin_total >= 63) & (a5 <= 105) & (chin_total >= 33):
                    # shutil.move(f, none_path)
                    if a4_R > a4_L:
                        a4 = round(angle3(landmark_tuple, 1, -2, 9))  # 왼쪽 사각턱 골격 각도
                        r_total = r_interesection(landmark_tuple, 7, -2, 21)
                    else:
                        a4 = round(angle3(landmark_tuple, 0, -3, 8))  # 오른쪽 사각턱 골격 각도
                        r_total = r_interesection(landmark_tuple, 6, -3, 20)
            # if R2 >= 1.45:
            #     shutil.move(f, long_path)
            # if (R2 < 1.45) & (a6 <= 71) & (a5 <= 88) & (r_total >= 260) & (abs(R5 - R6) <= 0.2) & (R5 <= 0.9) & (R6 <= 0.9):
            #     shutil.move(f, dia_path)
        result = [r_total, chin_total, lin_total, a4, a5, a6, R2, R5, R6]
        re = [min_max_normalize(result)]
        np.array(re)
        y_predict = model.predict(np.array(re))
        label = labels[int(y_predict[0])]
        y_predict = model.predict_proba(np.array(re))
        confidence = y_predict[0][y_predict[0].argmax()]

        if label == 'heart':
            shutil.copy2(f, heart_path)
            h_cnt += 1
        elif label == 'oval':
            shutil.copy2(f, oval_path)
            o_cnt += 1
        elif label == 'round':
            shutil.copy2(f, round_path)
            r_cnt += 1
        elif label == 'square':
            shutil.copy2(f, square_path)
            s_cnt += 1
        elif label == 'dia':
            shutil.copy2(f, square_path)
            d_cnt += 1
        elif label == 'long':
            shutil.copy2(f, square_path)
            l_cnt += 1
        else:
            shutil.copy2(f, none_path)
            n_cnt += 1
    except Exception as e:
        # print(img_name)
        pass
print(f'하트형 : {h_cnt}\n계란형 : {o_cnt}\n원형 : {r_cnt}\n사각형 : {s_cnt}\n마름모 : {d_cnt}\n긴형 : {l_cnt}')
# print(img_dir)

