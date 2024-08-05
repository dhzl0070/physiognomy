from face_shape_classify.utils.common import *
from face_shape_classify.utils.eyeshape_utils import *
from tqdm import tqdm
import glob, os, shutil, joblib

# EyebrowShapeModel = joblib.load(r'D:\workplace\test\shape\ex\train2\model\eyebrow_shape_classifier_XGB.pkl')
# labels = ['arch', 'deep_arch', 'flat', 'up']

img_dir = r'D:\data\yolo data\9label\images\HF\male'
home_path = r'D:\workplace\test\shape\test\valid'
result_path = home_path + r'\result'
nose_path = r'D:\workplace\test\shape\test'
big = nose_path + r'\big'
small = nose_path + r'\small'
long = nose_path + r'\long'
short = nose_path + r'\short'
none = nose_path + r'\none'

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
        landmark_nose = []
        labels = ['arch', 'deep_arch', 'flat', 'up']
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,  # True = 정적 이미지, False = 동적 이미지
                max_num_faces=1,  # 얼굴 최대 갯수
                refine_landmarks=True,  # 눈과 입술 주변의 랜드 마크 추가 출력 여부
                min_detection_confidence=0.5, ) as face_mesh:  # 신뢰도
            image = margin(iris_rotate(f))
            image_height, image_width, _ = image.shape
            face_direction_index = [2, 177, 401, 227, 447, 10, 152]
            nose_ft_index = [168, 1, 49, 279]
            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
            if results.multi_face_landmarks:
                annotated_image = image.copy()
                for single_face_landmarks in results.multi_face_landmarks:
                    for i in face_direction_index:
                        coordinates = single_face_landmarks.landmark[i]
                        if i == 10:  # 이마 가장자리 점
                            coordinates_8 = single_face_landmarks.landmark[8]
                            y_8 = coordinates_8.y * image_height  # 8번 y좌표 구하기
                            x = coordinates.x * image_width
                            y = (coordinates.y * image_height) - (
                                    (y_8 - (coordinates.y * image_height)) / 2)  # 10번에서 8까지 거리/2 만큼 10번 좌표 올리기
                        else:
                            x = coordinates.x * image_width
                            y = coordinates.y * image_height
                        face_direction_lst.append((x, y))
                        cv2.circle(annotated_image, (int(x), int(y)), 2, (0, 0, 255), -1)
                    for i in nose_ft_index:
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
            face_height = distance(face_direction_lst, 3, 4)
            face_width = distance(face_direction_lst, 5, 6)

            nose_height = distance(landmark_nose, 0, 1)
            nose_width = distance(landmark_nose, 2, 3)
            nose_ratio = round(nose_width / nose_height, 2)

            nose_height_ratio = round(nose_height / face_height, 2)
            nose_width_ratio = round(nose_width / face_width, 2)

            if (0.39 > nose_height_ratio >= 0.34) & (nose_width_ratio >= 0.22):
                shutil.copy2(f, big)
            elif (nose_height_ratio <= 0.32) & (nose_width_ratio <= 0.19):
                shutil.copy2(f, small)
            elif (nose_height_ratio >= 0.39):
                shutil.copy2(f, long)
            elif (nose_height_ratio <= 0.3):
                shutil.copy2(f, short)
            else:
                shutil.copy2(f, none)

    except Exception as e:
            # print(e)
            pass

print(f'arch : {a_cnt}, deep_arch : {da_cnt}, flat : {f_cnt}, up : {u_cnt}')