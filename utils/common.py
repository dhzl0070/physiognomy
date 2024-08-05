import cv2
import numpy as np
import collections
import math
from math import atan2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def iris_rotate(origin_img):
    landmark_iris = []  # 눈 중심점 landmarks
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,  # True = 정적 이미지, False = 동적 이미지
            max_num_faces=1,  # 얼굴 최대 갯수
            refine_landmarks=True,  # 눈과 입술 주변의 랜드마크 추가 출력 여부
            min_detection_confidence=0.5) as face_mesh:  # 신뢰도
        image = cv2.imread(origin_img)
        image_height, image_width, _ = image.shape
        NOSE_INDEX = [8, 2]
        # 작업 전에 BGR 이미지를 RGB로 변환합니다.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
        if results.multi_face_landmarks:
            iris_annotated_image = image.copy()
            for single_face_landmarks in results.multi_face_landmarks:
                # iris 좌표값 구하기
                for i in NOSE_INDEX:
                    coordinates = single_face_landmarks.landmark[i]
                    x = coordinates.x * image_width
                    y = coordinates.y * image_height
                    landmark_iris.append((x, y))

            left_eye_center = landmark_iris[0]
            left_eye_x = left_eye_center[0]
            left_eye_y = left_eye_center[1]
            right_eye_center = landmark_iris[1]
            right_eye_x = right_eye_center[0]
            right_eye_y = right_eye_center[1]

            if int(left_eye_x) < int(right_eye_x):
                point_3rd = (left_eye_x, right_eye_y)
                landmark_iris.append(point_3rd)
                direction = -1  # rotate same direction to clock
            else:
                point_3rd = (right_eye_x, left_eye_y)
                landmark_iris.append(point_3rd)
                direction = 1  # rotate inverse direction of clock

            angle = angle3(landmark_iris, 1, 0, 2)
            if direction == 1:
                angle = 90 - angle

            from PIL import Image
            new_img = Image.fromarray(iris_annotated_image)
            rotate_img = np.array(new_img.rotate(direction * angle))

            landmarks = []  # 얼굴 윤곽선 landmarks
            with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5) as face_mesh:
                image = rotate_img
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                new_output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_height, image_width, _ = new_output.shape
                INDEX = [10, 447, 152, 227]
                # 작업 전에 BGR 이미지를 RGB로 변환합니다.
                results = face_mesh.process(cv2.cvtColor(new_output, cv2.COLOR_BGR2RGB))
                # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
                if results.multi_face_landmarks:
                    landmark_annotated_image = new_output.copy()
                    for single_face_landmarks in results.multi_face_landmarks:
                        for i in INDEX:
                            coordinates = single_face_landmarks.landmark[i]
                            if i == 10:
                                coordinates_8 = single_face_landmarks.landmark[8]
                                y_8 = coordinates_8.y * image_height
                                x = coordinates.x * image_width
                                y = coordinates.y * image_height - (y_8 - (coordinates.y * image_height)) / 2
                            else:
                                x = coordinates.x * image_width
                                y = coordinates.y * image_height
                            landmarks.append((x, y))
                else:
                    print("landmarks not found")

            box = []
            box_check = []
            box_dict = {}
            x, y = interesection(0, landmarks[0][1], 640, landmarks[0][1], landmarks[1][0], 0, landmarks[1][0], 640)
            x2, y2 = interesection(0, landmarks[0][1], 640, landmarks[0][1], landmarks[3][0], 0, landmarks[3][0], 640)
            x3, y3 = interesection(0, landmarks[2][1], 640, landmarks[2][1], landmarks[1][0], 0, landmarks[1][0], 640)
            x4, y4 = interesection(0, landmarks[2][1], 640, landmarks[2][1], landmarks[3][0], 0, landmarks[3][0], 640)
            box.append((x + 10, y - 10))
            box.append((x2 - 10, y2 - 10))
            box.append((x3 + 10, y3 + 10))
            box.append((x4 - 10, y4 + 10))
            box_np = np.array(box, np.int32)
            x = round(box_np[1][0])
            y = round(box_np[1][1])
            w = round(box_np[0][0] - box_np[1][0])
            h = round(box_np[3][1] - box_np[1][1])
            box_dict["bbox"] = [x, y, w, h]

            for i in box_np:
                for j in i:
                    if j > 0:
                        box_check.append(True)
                    else:
                        box_check.append(False)
            if False not in box_check:
                img_trim = landmark_annotated_image[y:y + h, x:x + w]
            else:
                img_trim = [0]
    return img_trim

# 이미지 0 마진 처리
def margin(img):
    # 가로, 세로에 대해 부족한 margin 계산
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = input_img.shape[0:2]
    margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]

    # 부족한 길이가 절반으로 안 떨어질 경우 +1
    if np.abs(height - width) % 2 != 0:
        margin[0] += 1

    # 가로, 세로 가운데 부족한 쪽에 margin 추가
    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]

    # color 이미지일 경우 color 채널 margin 추가
    if len(input_img.shape) == 3:
        margin_list.append([0, 0])

    # 이미지에 margin 추가
    output = np.pad(input_img, margin_list, mode='constant')
    output = cv2.resize(output, (640, 640))

    # #save
    new_output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return new_output

# collections.nametuple을 이용하여 튜플을 리스트로 하나씩 담아주는 함수
Point2D = collections.namedtuple('Point2D', ['x', 'y'])

# 2선분의 교차점을 구하는 함수
def interesection(int1, int2, int3, int4, int5, int6, int7, int8):
    P1 = Point2D(int1, int2)
    P2 = Point2D(int3, int4)
    P3 = Point2D(int5, int6)
    P4 = Point2D(int7, int8)
    x1, y1 = P1.x, P1.y
    x2, y2 = P2.x, P2.y
    x3, y3 = P3.x, P3.y
    x4, y4 = P4.x, P4.y
    px = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    py = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    p = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    x = px / p
    y = py / p
    return x, y

# atan2로 3개의 선 사이각 구하는 함수
def angle3(landmark_tuple, int1, int2, int3):
    P0 = Point2D((landmark_tuple[int2][0]), (landmark_tuple[int2][1]))
    P1 = Point2D((landmark_tuple[int1][0]), (landmark_tuple[int1][1]))
    P2 = Point2D((landmark_tuple[int3][0]), (landmark_tuple[int3][1]))
    rad = atan2(P2.y - P0.y, P2.x - P0.x) - atan2(P1.y - P0.y, P1.x - P0.x)
    results = rad * (180 / math.pi)
    if results > 180:
        results -= 360
    return abs(results)  # abs 절대값

# 2개의 좌표 길이 구하는 함수
def distance(landmark_tuple, int1, int2):
    P1 = Point2D(int(landmark_tuple[int1][0]), int(landmark_tuple[int1][1]))
    P2 = Point2D(int(landmark_tuple[int2][0]), int(landmark_tuple[int2][1]))
    a = P1.x - P2.x
    b = P1.y - P2.y
    result = math.sqrt((a * a) + (b * b))
    return int(result)

# 기울기
def lin(landmark_tuple, int1, int2):
    P1 = Point2D(landmark_tuple[int1][0], landmark_tuple[int1][1])
    P2 = Point2D(landmark_tuple[int2][0], landmark_tuple[int2][1])
    slope = (P2.y - P1.y) / (P2.x - P1.x)
    return abs(slope)