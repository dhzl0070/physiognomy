from face_shape_classify.main_func import *
import shutil
import glob, os
from tqdm import tqdm
import pandas as pd

face_shape_name = 'square'
if face_shape_name == 'heart':
    face_shape = 0
elif face_shape_name == "oval":
    face_shape = 1
elif face_shape_name == 'round':
    face_shape = 2
elif face_shape_name == 'square':
    face_shape = 3
elif face_shape_name == 'dia':
    face_shape = 4
elif face_shape_name == 'long':
    face_shape = 5

img_dir = rf'D:\workplace\face_labeling\origin_img\{face_shape_name}'
home_path = r'D:\workplace'
none_path = home_path + r'\none'
ex_path = home_path + r'\ex'

colum_names = ['curvature_radius', 'chin_angle_inclease', 'chin_gradient', 'chin_born_angle', 'chin_shape_angle',
               'brow_shape_angle', 'ratio_width_height', 'raito_brow_chin', 'ratio_width_chin', 'face shape']

df = pd.DataFrame(columns=colum_names)
result_lst = []

cnt = 0
for f in tqdm(glob.glob(os.path.join(img_dir, "*.*"))):
    try:
        # 이미지 파일의 경우을 사용하세요.:
        img = f
        img_name = img.split(".")[0].split("\\")[-1]
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
        a4_R = round(angle3(landmark_tuple, 22, 18, 8))  # 턱 골격 각도
        a4_L = round(angle3(landmark_tuple, 23, 19, 9))  # 턱 골격 각도
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
        #         print("sec angle")
        #         if a4_R > a4_L:
        #             a4 = round(angle3(landmark_tuple, 1, -2, 9))  # 왼쪽 사각턱 골격 각도
        #             r_total = r_interesection(landmark_tuple, 7, -2, 21)
        #         else:
        #             a4 = round(angle3(landmark_tuple, 0, -3, 8))  # 오른쪽 사각턱 골격 각도
        #             r_total = r_interesection(landmark_tuple, 6, -3, 20)
        re = [r_total, chin_total, lin_total, a4, a5, a6, R2, R5, R6]
        result_lst.append((r_total, chin_total, lin_total, a4, a5, a6, R2, R5, R6, face_shape))
        for i in range(len(result_lst)):
            df.loc[i] = result_lst[i]
    except Exception as e:
        # shutil.move(f, none_path)
        pass
df.to_pickle(f'{ex_path}/{face_shape_name}.pkl')
