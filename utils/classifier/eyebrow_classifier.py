from face_shape_classify.utils.common import *
from face_shape_classify.utils.eyeshape_utils import *
from tqdm import tqdm
import glob, os, shutil, joblib

EyebrowShapeModel = joblib.load(r'D:\workplace\test\shape\ex\train2\model\eyebrow_shape_classifier_XGB.pkl')
labels = ['arch', 'deep_arch', 'flat', 'up']

img_dir = r'D:\workplace\test\shape\test\test'
home_path = r'D:\workplace\test\shape\test\valid'
result_path = home_path + r'\result'
eyebrows_path = r'D:\workplace\test\shape\test'
shape_round_path = result_path + r'\shape\round'
shape_big_path = result_path + r'\shape\big'
shape_normal_path = result_path + r'\shape\normal'
shape_small_path = result_path + r'\shape\small'
shape_thin_path = result_path + r'\shape\thin'
lin_down_path = result_path + r'\lin\down'
lin_normal_path = result_path + r'\lin\normal'
lin_up_path = result_path + r'\lin\up'
d_distant_path = result_path + r'\distance\distant'
d_normal_path = result_path + r'\distance\normal'
d_close_path = result_path + r'\distance\close'
arch_path = eyebrows_path + r'\arch'
deep_arch_path = eyebrows_path + r'\deep_arch'
none_path = eyebrows_path + r'\none'
flat_path = eyebrows_path + r'\flat'
up_path = eyebrows_path + r'\up'

a_cnt = 0
da_cnt = 0
f_cnt = 0
u_cnt = 0

for f in tqdm(glob.glob(os.path.join(img_dir, "*.*"))):
    try:
        # 이미지 파일의 경우을 사용하세요.:
        img_name = f.split(".")[0].split("\\")[-1]
        eye_shape = {}
        face_direction_lst = []
        landmark_iris = [] # 눈 중심점 landmarks
        landmark_left_eye = [] # 눈 좌측 landmarks
        landmark_right_eye = [] # 눈 우측 landmarks
        landmark_left_eyebrow = [] # 눈 좌측눈썹 landmarks
        landmark_right_eyebrow = [] # 눈 우측눈썹 landmarks
        landmark_eye_ratio = []
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,  # True = 정적 이미지, False = 동적 이미지
                max_num_faces=1,  # 얼굴 최대 갯수
                refine_landmarks=True,  # 눈과 입술 주변의 랜드마크 추가 출력 여부
                min_detection_confidence=0.5,) as face_mesh:  # 신뢰도
            image = margin(iris_rotate(f))
            image_height, image_width, _ = image.shape
            # iris 인덱스 번호(눈 바깥쪽에서 안쪽으로)
            left_eye_index = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
            right_eye_index = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
            left_eyebrow_index = [336, 296, 334, 293, 300]
            right_eyebrow_index = [107, 66, 105, 63, 70]
            face_direction_index = [2, 177, 401, 175]
            FT_INDEX = [227, 447, 33, 133, 362, 263]
            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
            if results.multi_face_landmarks:
                eye_annotated_image = image.copy()
                for single_face_landmarks in results.multi_face_landmarks:
                    for i in FT_INDEX:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_eye_ratio.append((x, y))
                    for i in face_direction_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        face_direction_lst.append((x, y))
                    for i in left_eye_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_left_eye.append((x, y))
                    for i in right_eye_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_right_eye.append((x, y))
                    for i in left_eyebrow_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_left_eyebrow.append((x, y))
                    for i in right_eyebrow_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_right_eyebrow.append((x, y))
        left_face_direction = int(face_direction_lst[2][0] - face_direction_lst[0][0])
        right_face_direction = int(face_direction_lst[0][0] - face_direction_lst[1][0])
        chin_face_direction = int(face_direction_lst[3][1] - face_direction_lst[0][1])
        # if chin_face_direction <= 145:
        #     shutil.copy2(f, arch_path)
        face_direction = abs(right_face_direction - left_face_direction)
        if face_direction < 100:
            eye_closed = int(landmark_left_eye[12][1]) - int(landmark_left_eye[5][1])
            if eye_closed > 15:
                if interesection_3P(landmark_left_eye, 0, 8, 4) > interesection_3P(landmark_left_eye, 0, 8, 5):
                    eye_uh = interesection_3P(landmark_left_eye, 0, 8, 4)
                else:
                    eye_uh = interesection_3P(landmark_left_eye, 0, 8, 5)

                if interesection_3P(landmark_left_eye, 0, 8, 11) > interesection_3P(landmark_left_eye, 0, 8, 12):
                    eye_dh = interesection_3P(landmark_left_eye, 0, 8, 11)
                else:
                    eye_dh = interesection_3P(landmark_left_eye, 0, 8, 12)

                eye_height = eye_uh + eye_dh
                eye_left_width = distance(landmark_left_eye, 0, 8)
                eye_right_width = distance(landmark_right_eye, 0, 8)
                eye_aspect_ratio = round(eye_left_width / eye_height, 2)  # 종횡비
                if eye_aspect_ratio <= 2.5:
                    eye_shape["eye_shape"] = 0
                    # shutil.copy2(f, shape_round_path)
                elif 2.5 < eye_aspect_ratio <= 2.75:
                    eye_shape["eye_shape"] = 1
                    # shutil.copy2(f, shape_big_path)
                elif 2.75 < eye_aspect_ratio <= 3.3:
                    eye_shape["eye_shape"] = 2
                    # shutil.copy2(f, shape_normal_path)
                elif 3.3 < eye_aspect_ratio <= 3.5:
                    eye_shape["eye_shape"] = 3
                    # shutil.copy2(f, shape_small_path)
                elif 3.5 < eye_aspect_ratio:
                    eye_shape["eye_shape"] = 4
                    # shutil.copy2(f, shape_thin_path)
                else:
                    eye_shape["eye_shape"] = None
                eye_shape_labels = ["round eye", "big eye", "normal eye", "small eye", "thin eyes"]

                a_angle = round(angle3(landmark_left_eye, 4, 0, 8), 2)
                b_angle = round(angle3(landmark_left_eye, 8, 0, 12), 2)
                angle_ratio = round(a_angle / b_angle, 2)
                eye_lin = round(lin(landmark_left_eye, 8, 0), 2)

                if eye_lin <= 0.1:
                    eye_shape["eye_lin"] = 0
                    # shutil.copy2(f, lin_down_path)
                elif 0.1 < eye_lin < 0.2:
                    eye_shape["eye_lin"] = 1
                    # shutil.copy2(f, lin_normal_path)
                elif 0.2 <= eye_lin:
                    eye_shape["eye_lin"] = 2
                    # shutil.copy2(f, lin_up_path)
                else:
                    eye_shape["eye_lin"] = None
                eye_lin_labels = ["down", "normal", "up"]

                landmark_eye_ratio_lst = []
                landmark_eye_ratio_lst.append(landmark_eye_ratio[0])
                landmark_eye_ratio_lst.append(landmark_eye_ratio[1])
                for i in range(2, 6, 1):
                    x, y = interesection2(landmark_eye_ratio, 0, 1, i)
                    landmark_eye_ratio_lst.append((x, y))

                # 가로 5등분 비율 계산
                d1 = distance(landmark_eye_ratio_lst, 0, 2)
                d2 = distance(landmark_eye_ratio_lst, 2, 3)  # 오른쪽 눈
                d3 = distance(landmark_eye_ratio_lst, 3, 4)
                d4 = distance(landmark_eye_ratio_lst, 4, 5)  # 왼쪽 눈
                d5 = distance(landmark_eye_ratio_lst, 5, 1)
                A1 = distance(landmark_eye_ratio_lst, 0, 1)  # 얼굴 가로 길이

                if d2 > d4:
                    eye_width = d2
                else:
                    eye_width = d4

                eye_d_lst = [landmark_left_eye[8], landmark_right_eye[8]]
                eye_distance = distance(eye_d_lst, 0, 1)
                eye_distance_avg = round(((eye_left_width + eye_right_width) / 2), 2)
                d_ratio = round(d3 / A1, 2)

                if (eye_distance >= 125) & (d_ratio >= 0.28):
                    eye_shape["eye_distance"] = 0
                    # shutil.copy2(f, d_distant_path)
                elif (100 > eye_distance) & (0.22 >= d_ratio):
                    eye_shape["eye_distance"] = 2
                    # shutil.copy2(f, d_close_path)
                else:
                    eye_shape["eye_distance"] = 1
                    # shutil.copy2(f, d_normal_path)

                eye_distance_labels = ["distant", "normal", "close"]

                eye_shape_labels = eye_shape_labels[int(eye_shape["eye_shape"])]
                eye_shape["eye_shape"] = eye_shape_labels
                eye_lin_labels = eye_lin_labels[int(eye_shape["eye_lin"])]
                eye_shape["eye_lin"] = eye_lin_labels
                eye_distance_labels = eye_distance_labels[int(eye_shape["eye_distance"])]
                eye_shape["eye_distance"] = eye_distance_labels

                EBD = []
                eyebrow_lst = [(0, 8), (1, 7), (2, 6), (3, 5)]

                for r in range(0, 4, 1):
                    start = int(landmark_left_eyebrow[r][0]), int(landmark_left_eyebrow[r][1])
                    end = int(landmark_left_eyebrow[r + 1][0]), int(landmark_left_eyebrow[r + 1][1])
                    cv2.line(eye_annotated_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

                if landmark_left_eyebrow[0][1] > landmark_left_eyebrow[4][1]:
                    landmark_left_eyebrow.append(landmark_left_eyebrow[0])
                    landmark_left_eyebrow.append((landmark_left_eyebrow[4][0], landmark_left_eyebrow[0][1]))
                else:
                    landmark_left_eyebrow.append((landmark_left_eyebrow[0][0], landmark_left_eyebrow[4][1]))
                    landmark_left_eyebrow.append(landmark_left_eyebrow[4])
                EBD.append(distance(landmark_left_eyebrow, 5, 6))
                start = int(landmark_left_eyebrow[5][0]), int(landmark_left_eyebrow[5][1])
                end = int(landmark_left_eyebrow[6][0]), int(landmark_left_eyebrow[6][1])
                cv2.line(eye_annotated_image, start, end, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                for i in range(0, 5, 1):
                    landmark_left_eyebrow.append(interesection2(landmark_left_eyebrow, 5, 6, i))
                    EBD.append(distance(landmark_left_eyebrow, i, i + 7))

                # 기울기
                el0 = round(eyebrowlin(landmark_left_eyebrow, 0, 1), 2)
                el1 = round(eyebrowlin(landmark_left_eyebrow, 1, 2), 2)
                el2 = round(eyebrowlin(landmark_left_eyebrow, 2, 3), 2)
                el3 = round(eyebrowlin(landmark_left_eyebrow, 3, 4), 2)
                lm0 = abs(round((el1 - el0) / el0 * 100))
                lm1 = abs(round((el2 - el1) / el1 * 100))
                lm2 = abs(round((el3 - el2) / el2 * 100))
                # if (el0 >= 0.2) & (lm0 < 15) & ((EBD[2] - EBD[1]) >= 10):
                #     shutil.copy2(f, up_path)
                # elif (el0 <= 0.15) & (lm0 > 40) & ((EBD[2] - EBD[1]) < 10):
                #     # shutil.copy2(f, flat_path)
                #     pass
                # elif (el0 >= 0.19) & (lm0 > 40):
                #     if EBD[0] < 137:
                #         # shutil.copy2(f, deep_arch_path)
                #         pass
                #     else:
                #         # shutil.copy2(f, arch_path)
                #         pass
                # else:
                #     shutil.copy2(f, none_path)

                # 곡률 반경
                er = r_interesection(landmark_left_eyebrow, 0, 2, 4)
                result = [EBD[1], EBD[2], EBD[3], EBD[4], EBD[5],
                          el0, el1, el2, el3, lm0, lm1, lm2]
                re = [eyebrow_min_max_normalize(result)]
                np.array(re)
                y_predict = EyebrowShapeModel.predict(np.array(re))
                label = labels[int(y_predict[0])]
                y_predict = EyebrowShapeModel.predict_proba(np.array(re))
                confidence = y_predict[0][y_predict[0].argmax()]
                for i in range(len(y_predict[0])):
                    y_predict[0][i] = round(y_predict[0][i], 2) * 100
                if label == 'arch':
                    a_cnt += 1
                    # shutil.copy2(f, arch_path)
                    cv2.imwrite(f"{arch_path}/{img_name}.jpg", eye_annotated_image)
                elif label == 'deep_arch':
                    da_cnt += 1
                    # shutil.copy2(f, deep_arch_path)
                    cv2.imwrite(f"{deep_arch_path}/{img_name}.jpg", eye_annotated_image)
                elif label == 'flat':
                    f_cnt += 1
                    # shutil.copy2(f, flat_path)
                    cv2.imwrite(f"{flat_path}/{img_name}.jpg", eye_annotated_image)
                elif label == 'up':
                    u_cnt += 1
                    # shutil.copy2(f, up_path)
                    cv2.imwrite(f"{up_path}/{img_name}.jpg", eye_annotated_image)
                else:
                    # shutil.copy2(f, none_path)
                    pass
                # shutil.copy2(f, lin_down_path)
                # cv2.imwrite(f"{home_path}/{img_name}.jpg", eye_annotated_image)
    except Exception as e:
            # print(e)
            pass

print(f'arch : {a_cnt}, deep_arch : {da_cnt}, flat : {f_cnt}, up : {u_cnt}')