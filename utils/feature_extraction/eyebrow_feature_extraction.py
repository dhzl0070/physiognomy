from face_shape_classify.utils.common import *
from face_shape_classify.utils.eyeshape_utils import *
import glob, os, shutil
from tqdm import tqdm
import pandas as pd

eyebrow_shape_name = 'flat'
if eyebrow_shape_name == 'arch':
    eyebrow_shape = 0
elif eyebrow_shape_name == "deep_arch":
    eyebrow_shape = 1
elif eyebrow_shape_name == 'flat':
    eyebrow_shape = 2
elif eyebrow_shape_name == 'up':
    eyebrow_shape = 3

img_dir = rf'D:\workplace\test\shape\eyebrows\train2\{eyebrow_shape_name}'
home_path = r'D:\workplace\test\shape'
none_path = home_path + r'\none'
ex_path = home_path + r'\ex\train2'

colum_names = ['ef0', 'ef1', 'ef2', 'ef3','ef4', 'rad0', 'rad1', 'rad2', 'rad3'
               ,'rad_ratio0', 'rad_ratio1', 'rad_ratio2', 'eyebrow_shape']

df = pd.DataFrame(columns=colum_names)
result_lst = []
cnt = 0
for f in tqdm(glob.glob(os.path.join(img_dir, "*.*"))):
    try:
        # 이미지 파일의 경우을 사용하세요.:
        img_name = f.split(".")[0].split("\\")[-1]
        eye_shape = {}
        face_direction_lst = []
        landmark_iris = []  # 눈 중심점 landmarks
        landmark_left_eye = []  # 눈 좌측 landmarks
        landmark_right_eye = []  # 눈 우측 landmarks
        landmark_left_eyebrow = []  # 눈 좌측눈썹 landmarks
        landmark_right_eyebrow = []  # 눈 우측눈썹 landmarks
        landmark_eye_ratio = []
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,  # True = 정적 이미지, False = 동적 이미지
                max_num_faces=1,  # 얼굴 최대 갯수
                refine_landmarks=True,  # 눈과 입술 주변의 랜드마크 추가 출력 여부
                min_detection_confidence=0.5, ) as face_mesh:  # 신뢰도
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
                        cv2.circle(eye_annotated_image, (int(x), int(y)), 2, (0, 0, 255), -1)
                    for i in face_direction_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        face_direction_lst.append((x, y))
                        cv2.circle(eye_annotated_image, (int(x), int(y)), 2, (0, 0, 255), -1)
                    for i in left_eye_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_left_eye.append((x, y))
                        cv2.circle(eye_annotated_image, (int(x), int(y)), 2, (0, 0, 255), -1)
                    for i in right_eye_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_right_eye.append((x, y))
                        cv2.circle(eye_annotated_image, (int(x), int(y)), 2, (0, 0, 255), -1)
                    for i in left_eyebrow_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_left_eyebrow.append((x, y))
                        cv2.circle(eye_annotated_image, (int(x), int(y)), 2, (0, 0, 255), -1)
                    for i in right_eyebrow_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_right_eyebrow.append((x, y))
                        cv2.circle(eye_annotated_image, (int(x), int(y)), 2, (0, 0, 255), -1)
        left_face_direction = int(face_direction_lst[2][0] - face_direction_lst[0][0])
        right_face_direction = int(face_direction_lst[0][0] - face_direction_lst[1][0])
        chin_face_direction = int(face_direction_lst[3][1] - face_direction_lst[0][1])
        # if chin_face_direction <= 145:
        #     shutil.copy2(f, arch_path)
        face_direction = abs(right_face_direction - left_face_direction)
        if face_direction < 100:
            eye_closed = int(landmark_left_eye[12][1]) - int(landmark_left_eye[5][1])
            if eye_closed > 15:
                EBD = []
                eyebrow_lst = [(0, 8), (1, 7), (2, 6), (3, 5)]

                if landmark_left_eyebrow[0][1] > landmark_left_eyebrow[4][1]:
                    landmark_left_eyebrow.append(landmark_left_eyebrow[0])
                    landmark_left_eyebrow.append((landmark_left_eyebrow[4][0], landmark_left_eyebrow[0][1]))
                else:
                    landmark_left_eyebrow.append((landmark_left_eyebrow[0][0], landmark_left_eyebrow[4][1]))
                    landmark_left_eyebrow.append(landmark_left_eyebrow[4])
                EBD.append(distance(landmark_left_eyebrow, 5, 6))

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

                # 곡률 반경
                er = r_interesection(landmark_left_eyebrow, 0, 2, 4)

                result_lst.append((EBD[1], EBD[2], EBD[3], EBD[4], EBD[5],
                      el0, el1, el2, el3, lm0, lm1, lm2, eyebrow_shape))
                for i in range(len(result_lst)):
                    df.loc[i] = result_lst[i]
    except Exception as e:
        pass
        # shutil.move(f, none_path)
df.to_pickle(f'{ex_path}/{eyebrow_shape_name}.pkl')
