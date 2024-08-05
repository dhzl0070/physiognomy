from face_shape_classify.main_func import *
import shutil
import glob, os
from tqdm import tqdm
import pandas as pd
import joblib
import warnings
warnings.filterwarnings(action='ignore')

img_dir = r'D:\workplace\test\shape\valid_copy\faceshape\square_copy'
home_path = r'D:\workplace\round'
square_path = home_path + r'\square'
labels = ['heart', 'oval', 'round', 'square', 'dia', 'long']

for f in tqdm(glob.glob(os.path.join(img_dir, "*.*"))):
    try:
        # 이미지 파일의 경우을 사용하세요.:
        img = f
        img_name = img[0].split(".")[0].split("\\")[-1]

        img = margin(iris_rotate(img))
        landmarks_total_dict = Landmarks.landmarks(img, img_name)
        landmark_tuple = landmarks_total_dict['faceshape']

        # 턱 골격 교차점 좌표
        x, y = square_interesection(landmark_tuple, 22, 6, 20, 8)
        landmark_tuple.append((x, y))
        x, y = square_interesection(landmark_tuple, 23, 7, 21, 9)
        landmark_tuple.append((x, y))
        # 턱 모양 교차점 좌표
        x, y = square_interesection(landmark_tuple, 18, 10, 11, 19)
        landmark_tuple.append((x, y))
        a5 = round(angle3(landmark_tuple, 18, -1, 19))  # 턱의 모양 각도
        a6 = round(angle3(landmark_tuple, 0, 4, 1))  # 이마 사이각
        a7 = round(angle3(landmark_tuple, 2, 0, 6))  # 얼굴 모양 각도
        a4_R = round(angle3(landmark_tuple, 22, 18, 8))  # 오른쪽 턱 골격 각도
        a4_L = round(angle3(landmark_tuple, 23, 19, 9))  # 왼쪽 턱 골격 각도
        # 얼굴 길이
        A1 = distance(landmark_tuple, 0, 1)  # 얼굴 가로 길이
        A2 = distance(landmark_tuple, 2, 3)  # 얼굴 이마 길이
        A3 = distance(landmark_tuple, 4, 5)  # 얼굴 세로 길이
        A4 = distance(landmark_tuple, 6, 7)  # 얼굴 턱 사이 길이
        # 얼굴 비율
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
        # if R2 <= 1.3:
        #     if (lin_total >= 63) & (a5 <= 105) & (chin_total >= 33):
        #         if a4_R > a4_L:
        #             a4 = round(angle3(landmark_tuple, 1, -2, 9))  # 왼쪽 사각턱 골격 각도
        #             r_total = r_interesection(landmark_tuple, 7, -2, 21)
        #         else:
        #             a4 = round(angle3(landmark_tuple, 0, -3, 8))  # 오른쪽 사각턱 골격 각도
        #             r_total = r_interesection(landmark_tuple, 6, -3, 20)

        top_dict = {}
        result = [r_total, chin_total, lin_total, a4, a5, a6, R2, R5, R6]
        re = [min_max_normalize(result)]
        y_predict = FaceShapeModel.predict(np.array(re))
        label = labels[int(y_predict[0])]
        y_predict = FaceShapeModel.predict_proba(np.array(re))
        confidence = y_predict[0][y_predict[0].argmax()]
        for i in range(len(y_predict[0])):
            y_predict[0][i] = round(y_predict[0][i], 2) * 100
            top_dict[y_predict[0][i]] = labels[i]
        faceshape_result = {"label": label, "confidence": round((confidence * 100), 2)}
        # 순위 뽑기
        sort_lst = sorted(y_predict[0], reverse=True)
        sort_list = f'{top_dict[sort_lst[0]]} - {int(sort_lst[0])}%, {top_dict[sort_lst[1]]} - {int(sort_lst[1])}%'
        result_1 = faceshape_result["label"]
        if result_1 == 'square':
            shutil.copy2(f, square_path)
        else:
            pass
    except Exception as e:
        # print(img_name)
        pass

