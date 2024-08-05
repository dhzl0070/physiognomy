from face_shape_classify.utils.common import *
from face_shape_classify.utils.eyeshape_utils import *
from tqdm import tqdm
import glob, os, shutil, joblib

# EyebrowShapeModel = joblib.load(r'D:\workplace\test\shape\ex\train2\model\eyebrow_shape_classifier_XGB.pkl')
# labels = ['arch', 'deep_arch', 'flat', 'up']

img_dir = r'D:\data\yolo data\9label\images\HF\female'
home_path = r'D:\workplace\test\shape\test\valid'
result_path = home_path + r'\result'
mouth_path = r'D:\workplace\test\shape\test'
big = mouth_path + r'\big'
small = mouth_path + r'\small'
upperthick = mouth_path + r'\upperthick'
lowerthick = mouth_path + r'\lowerthick'
thick = mouth_path + r'\thick'
thin = mouth_path + r'\thin'
none = mouth_path + r'\none'

a_cnt = 0
da_cnt = 0
f_cnt = 0
u_cnt = 0


for f in tqdm(glob.glob(os.path.join(img_dir, "*.*"))):
    try:
        # 이미지 파일의 경우을 사용하세요.:
        img_name = f[0].split(".")[0].split("\\")[-1]
        mouth_shape = {}
        face_direction_lst = []
        landmark_mouth = []
        landmark_nose = []
        labels = ['arch', 'deep_arch', 'flat', 'up']
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,  # True = 정적 이미지, False = 동적 이미지
                max_num_faces=1,  # 얼굴 최대 갯수
                refine_landmarks=True,  # 눈과 입술 주변의 랜드 마크 추가 출력 여부
                min_detection_confidence=0.5, ) as face_mesh:  # 신뢰도
            image = margin(iris_rotate(f))
            image_height, image_width, _ = image.shape
            face_direction_index = [2, 177, 401, 152]
            mouth_index = [0, 13, 14, 17, 61, 291, 267, 312]
            nose_index = [49, 279]
            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
            if results.multi_face_landmarks:
                annotated_image = image.copy()
                for single_face_landmarks in results.multi_face_landmarks:
                    for i in face_direction_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        face_direction_lst.append((x, y))
                        cv2.circle(annotated_image, (int(x), int(y)), 2, (0, 0, 255), -1)
                    for i in mouth_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_mouth.append((x, y))
                        cv2.circle(annotated_image, (int(x), int(y)), 2, (0, 0, 255), -1)
                    for i in nose_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_nose.append((x, y))
                        cv2.circle(annotated_image, (int(x), int(y)), 2, (0, 0, 255), -1)

        left_face_direction = int(face_direction_lst[2][0] - face_direction_lst[0][0])
        right_face_direction = int(face_direction_lst[0][0] - face_direction_lst[1][0])
        chin_face_direction = int(face_direction_lst[3][1] - face_direction_lst[0][1])
        face_direction = abs(right_face_direction - left_face_direction)
        if face_direction < 100:
            # index 7 x 좌표 index 6 x 좌표로 수정
            landmark_mouth[7] = (landmark_mouth[6][0], landmark_mouth[7][1])
            philtrum = []
            philtrum.append(face_direction_lst[0])
            philtrum.append(landmark_mouth[0])
            # 턱 너비
            chin = distance(face_direction_lst, 1, 2)
            # 인중 거리
            philtrum_distance = distance(philtrum, 0, 1)
            # 코 너비
            nose_width = distance(landmark_nose, 0, 1)
            # 입 너비
            mouth_width = distance(landmark_mouth, 4, 5)
            upper_mouth_height = distance(landmark_mouth, 6, 7)
            lower_mouth_height = distance(landmark_mouth, 2, 3)
            mouth_height = upper_mouth_height + lower_mouth_height

            nose_mouth_ratio = round(mouth_width / nose_width, 2)
            mouth_ratio = round(mouth_width / mouth_height, 2)
            chin_mouth_ratio = round(mouth_width / chin, 2)
            lip_ratio = round((lower_mouth_height - upper_mouth_height) / upper_mouth_height * 100, 2)

            # if chin_mouth_ratio < 0.32:
            #     shutil.copy2(f, mouth_path)
            #     # cv2.imwrite(f"{home_path}/{img_name}.jpg", eye_annotated_image)
            if (nose_mouth_ratio >= 1.45) & (chin_mouth_ratio >= 0.45):
                shutil.copy2(f, big)
            elif (nose_mouth_ratio <= 1.2) & (chin_mouth_ratio <= 0.33):
                shutil.copy2(f, small)
            elif (upper_mouth_height > lower_mouth_height) & (lip_ratio < -5) & (upper_mouth_height > 30):
                shutil.copy2(f, upperthick)
            elif (upper_mouth_height < lower_mouth_height) & (lip_ratio > 50) & (lower_mouth_height > 30):
                shutil.copy2(f, lowerthick)
            elif (upper_mouth_height >= 30) & (lower_mouth_height >= 32):
                shutil.copy2(f, thick)
            elif (upper_mouth_height <= 26) & (lower_mouth_height <= 26):
                shutil.copy2(f, thin)
            else:
                shutil.copy2(f, none)
    except Exception as e:
            # print(e)
            pass

print(f'arch : {a_cnt}, deep_arch : {da_cnt}, flat : {f_cnt}, up : {u_cnt}')