from face_shape_classify.utils.common import *
from face_shape_classify.utils.drawline import *
from face_shape_classify.utils.faceshape_utils import *
from face_shape_classify.utils.eyeshape_utils import *
from face_shape_classify.utils.mouthshape_utils import *
from face_shape_classify.utils.landmark_index import *
import joblib
import warnings
warnings.filterwarnings(action='ignore')

FaceShapeModel = joblib.load(r'D:\workplace\test\shape\ex\train2\model\face_shape_classifier_XGB_final.pkl')
# FaceShapeModel = joblib.load(r'D:\workplace\ex\model\face_shape_classifier_XGB.pkl')
EyebrowShapeModel = joblib.load(r'D:\workplace\test\shape\ex\train2\model\eyebrow_shape_3_classifier_XGB.pkl')

home_path = r'D:\workplace\test\shape\img_save'

class Landmarks:
    def __init__(self):
        super(Landmarks).__init__()
    def landmarks(rotate_img):
        landmarks_total_dict = {}
        # 얼굴 수평 방향
        face_direction_lst = []
        face_contour_lst = []
        # 얼굴 랜드마크
        faceshape_landmark_tuple = []
        # 눈 랜드마크
        landmark_left_eye = []
        landmark_right_eye = []
        # 눈썹 랜드마크
        landmark_left_eyebrow = []
        landmark_right_eyebrow = []
        landmark_eye_ratio = []
        # 입 랜드마크
        landmark_mouth = []
        mouth_contour_lst = []
        # 코 랜드마크
        landmark_face = []
        landmark_nose = []
        nose_contour_lst = []
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            image_height, image_width, _ = rotate_img.shape
            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            results = face_mesh.process(cv2.cvtColor(rotate_img, cv2.COLOR_BGR2RGB))
            # 이미지에 출력하고 그 위에 얼굴 그물망 경계점을 그립니다.
            if results.multi_face_landmarks:
                face_image = rotate_img.copy()
                face_draw_image = rotate_img.copy()
                eye_image = rotate_img.copy()
                eye_draw_image = rotate_img.copy()
                mouth_image = rotate_img.copy()
                mouth_draw_image = rotate_img.copy()
                nose_image = rotate_img.copy()
                nose_draw_image = rotate_img.copy()
                total_image = rotate_img.copy()
                total_draw_image = rotate_img.copy()
                for single_face_landmarks in results.multi_face_landmarks:
                    def draw_point(point):
                        coordinate = single_face_landmarks.landmark[point]
                        y_point = coordinate.y * image_height
                        x = coordinates.x * image_width
                        y = (coordinates.y * image_height) - (
                                (y_point - (coordinates.y * image_height)) / 2)  # 10번에서 8까지 거리/2 만큼 10번 좌표 올리기
                        return x, y
                    # 좌표 추출
                    for i in face_direction_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        z = coordinates.z
                        face_direction_lst.append((x, y, z))
                    for i in face_contour_index:
                        coordinates = single_face_landmarks.landmark[i]
                        if i == 10:  # 이마 가장자리 점
                            x, y = draw_point(8)
                        elif i == 109:  # 이마 가장자리 점
                            x, y = draw_point(55)
                        elif i == 338:  # 이마 가장자리 점
                            x, y = draw_point(285)
                        elif i == 67:  # 이마 가장자리 점
                            x, y = draw_point(65)
                        elif i == 297:  # 이마 가장자리 점
                            x, y = draw_point(295)
                        elif i == 103:  # 이마 가장자리 점
                            x, y = draw_point(63)
                        elif i == 332:  # 이마 가장자리 점
                            x, y = draw_point(293)
                        elif i == 54:  # 이마 가장자리 점
                            x, y = draw_point(68)
                        elif i == 284:  # 이마 가장자리 점
                            x, y = draw_point(298)
                        else:
                            x = coordinates.x * image_width
                            y = coordinates.y * image_height
                        cv2.circle(face_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        cv2.circle(total_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        cv2.putText(face_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 0), 1)
                        # cv2.putText(total_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                        face_contour_lst.append((x, y))
                    for i in faceshape_landmark_index:
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
                        faceshape_landmark_tuple.append((x, y))
                    for i in FT_INDEX:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_eye_ratio.append((x, y))
                    for i in left_eye_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_left_eye.append((x, y))
                        cv2.circle(eye_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(eye_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                        cv2.circle(total_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(total_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                    for i in right_eye_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_right_eye.append((x, y))
                        cv2.circle(eye_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(eye_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                        cv2.circle(total_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(total_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                    for i in left_eyebrow_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_left_eyebrow.append((x, y))
                        cv2.circle(eye_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(eye_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                        cv2.circle(total_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(total_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                    for i in right_eyebrow_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_right_eyebrow.append((x, y))
                        cv2.circle(eye_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(eye_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                        cv2.circle(total_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(total_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                    for i in mouth_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_mouth.append((x, y))
                    for i in mouth_contour_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        mouth_contour_lst.append((x, y))
                        cv2.circle(mouth_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(mouth_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                        cv2.circle(total_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(total_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                    for i in face_index:
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
                        landmark_face.append((x, y))
                    for i in nose_ft_index:
                        coordinates = single_face_landmarks.landmark[i]
                        x = coordinates.x * image_width
                        y = coordinates.y * image_height
                        landmark_nose.append((x, y))
                        cv2.circle(nose_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(nose_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                        cv2.circle(total_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(total_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                    for i in nose_index:
                        coordinates = single_face_landmarks.landmark[i]
                        if i == 245:  # 이마 가장자리 점
                            x = coordinates.x * image_width + 5
                            y = coordinates.y * image_height
                        elif i == 465:  # 이마 가장자리 점
                            x = coordinates.x * image_width - 5
                            y = coordinates.y * image_height
                        else:
                            x = coordinates.x * image_width
                            y = coordinates.y * image_height
                        cv2.circle(nose_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(nose_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                        cv2.circle(total_image, (int(x), int(y)), 4, (0, 0, 255), -1)
                        # cv2.putText(total_image, str(i), (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #             (0, 0, 0), 1)
                        nose_contour_lst.append((x,y))
                # cv2.imshow("test", face_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        # 얼굴 수평 방향
        left_face_direction = int(face_direction_lst[2][0] - face_direction_lst[0][0])
        right_face_direction = int(face_direction_lst[0][0] - face_direction_lst[1][0])
        face_direction = abs(right_face_direction - left_face_direction)
        # 얼굴 수직 방향
        hp = round(face_direction_lst[4][2], 2)
        tp = round(face_direction_lst[3][2], 2)
        # 눈 닫힘 확인
        eye_closed = int(landmark_left_eye[12][1]) - int(landmark_left_eye[5][1])
        # 입 닫힘 확인
        mouth_close = distance(landmark_mouth, 1, 2)
        # 입 꼬리 선 그리기
        start = int(landmark_mouth[9][0]), int(landmark_mouth[9][1])
        end = int(landmark_mouth[5][0]), int(landmark_mouth[5][1])
        cv2.line(mouth_image, start, end, (0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
        start = int(landmark_mouth[8][0]), int(landmark_mouth[8][1])
        end = int(landmark_mouth[4][0]), int(landmark_mouth[4][1])
        cv2.line(mouth_image, start, end, (0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

        # & (-0.14 < hp <= 0.02) & (-0.02 < tp <= 0.12)
        if (face_direction <= 100) & (eye_closed > 15) & (mouth_close < 15) & (-0.14 < hp <= 0.02) & (-0.02 < tp <= 0.12) :
            landmarks_total_dict = {"faceshape" : faceshape_landmark_tuple,
                                    "eyeshape": [landmark_eye_ratio, landmark_left_eye, landmark_right_eye, landmark_left_eyebrow],
                                    "mouthshape": [face_direction_lst, landmark_mouth, landmark_nose],
                                    "noseshape": [landmark_face, landmark_nose]
                                    }
            # if save_check == True:
            #     face_draw_image = draw_face(face_contour_lst, face_draw_image)
            #     total_draw_image = draw_face(face_contour_lst, total_draw_image)
            #     eye_draw_image = draw_eye(landmark_left_eye, landmark_right_eye, eye_draw_image)
            #     total_draw_image = draw_eye(landmark_left_eye, landmark_right_eye, total_draw_image)
            #     eye_draw_image = draw_eyebrow(landmark_left_eyebrow, landmark_right_eyebrow, eye_draw_image)
            #     total_draw_image = draw_eyebrow(landmark_left_eyebrow, landmark_right_eyebrow, total_draw_image)
            #     nose_draw_image = draw_nose(nose_contour_lst, nose_draw_image)
            #     total_draw_image = draw_nose(nose_contour_lst, total_draw_image)
            #     mouth_draw_image = draw_mouth(mouth_contour_lst, mouth_draw_image)
            #     total_draw_image = draw_mouth(mouth_contour_lst, total_draw_image)
            #     cv2.imwrite(f"{home_path}/{img_name}_face_image.jpg", face_image)
            #     cv2.imwrite(f"{home_path}/{img_name}_face_draw_image.jpg", face_draw_image)
            #     cv2.imwrite(f"{home_path}/{img_name}_eye_image.jpg", eye_image)
            #     cv2.imwrite(f"{home_path}/{img_name}_eye_draw_image.jpg", eye_draw_image)
            #     cv2.imwrite(f"{home_path}/{img_name}_mouth_image.jpg", mouth_image)
            #     cv2.imwrite(f"{home_path}/{img_name}_mouth_draw_image.jpg", mouth_draw_image)
            #     cv2.imwrite(f"{home_path}/{img_name}_nose_image.jpg", nose_image)
            #     cv2.imwrite(f"{home_path}/{img_name}_nose_draw_image.jpg", nose_draw_image)
            #     cv2.imwrite(f"{home_path}/{img_name}_total_image.jpg", total_image)
            #     cv2.imwrite(f"{home_path}/{img_name}_total_draw_image.jpg", total_draw_image)
        else:
            print(f'Rotate face_direction : {face_direction <= 100}, '
                  f'eye_closed : {eye_closed > 15}, '
                  f'mouth_close : {mouth_close < 10}')
            landmarks_total_dict = None
        return landmarks_total_dict
class FaceShape:
    def __init__(self):
        super(FaceShape).__init__()
    def faceshape(landmark_dict):
        faceshape_result = {}
        # labels = ['heart', 'oval', 'round', 'square', 'dia', 'long']
        labels = [
                  '하트형 얼굴',
                  '계란형 얼굴',
                  '원형 얼굴',
                  '사각형 얼굴',
                  '마름모형 얼굴',
                  '타원형 얼굴'
                  ]
        landmark_tuple = landmark_dict['faceshape']

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

        if R2 <= 1.3:
            if (lin_total >= 63) & (a5 <= 105) & (chin_total >= 33):
                if a4_R > a4_L:
                    a4 = round(angle3(landmark_tuple, 1, -2, 9))  # 왼쪽 사각턱 골격 각도
                    r_total = r_interesection(landmark_tuple, 7, -2, 21)
                else:
                    a4 = round(angle3(landmark_tuple, 0, -3, 8))  # 오른쪽 사각턱 골격 각도
                    r_total = r_interesection(landmark_tuple, 6, -3, 20)

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
        faceshape_result = {"label":label, "confidence":round((confidence * 100), 2)}
        # 순위 뽑기
        sort_list = sorted(y_predict[0], reverse=True)
        sort_list = f'{top_dict[sort_list[0]]} - {int(sort_list[0])}%, {top_dict[sort_list[1]]} - {int(sort_list[1])}%'

        return faceshape_result, sort_list
class EyeShape:
    def __init__(self):
        super(EyeShape).__init__()
    def eyeshape(landmark_dict):
        eyeshape_result = {}
        eyebrowshape_result = {}
        eye_shape = {}
        landmark_eye_ratio = landmark_dict["eyeshape"][0]
        landmark_left_eye = landmark_dict["eyeshape"][1]  # 눈 좌측 landmarks
        landmark_right_eye = landmark_dict["eyeshape"][2]  # 눈 우측 landmarks
        landmark_left_eyebrow = landmark_dict["eyeshape"][3]  # 눈 좌측 눈썹 landmarks
        # landmark_right_eyebrow = []  # 눈 우측 눈썹 landmarks
        # eye_shape_labels = ["round", "big", "normal", "small", "thin"]
        eye_shape_labels = ["동그란 눈", "큰 눈", "보통 눈", "작은 눈", "가는 눈"]
        # eye_lin_labels = ["down", "normal", "up"]
        eye_lin_labels = ["눈 끝이 위로 올라간", "눈 끝이 일자인", "눈 끝이 아래로 쳐진"]
        # eye_distance_labels = ["distant", "normal", "close"]
        eye_distance_labels = ["눈 사이가 넓은", "눈 사이가 보통", "눈 사이가 좁은"]
        # eyebrow_labels = ['arch', 'flat', 'up']
        eyebrow_labels = ['아치형 눈썹', '일자 눈썹', '올라간 눈썹']

        # 눈 상단 높이
        if interesection_3P(landmark_left_eye, 0, 8, 4) > interesection_3P(landmark_left_eye, 0, 8, 5):
            eye_uh = interesection_3P(landmark_left_eye, 0, 8, 4)
        else:
            eye_uh = interesection_3P(landmark_left_eye, 0, 8, 5)

        # 눈 하단 높이
        if interesection_3P(landmark_left_eye, 0, 8, 11) > interesection_3P(landmark_left_eye, 0, 8, 12):
            eye_dh = interesection_3P(landmark_left_eye, 0, 8, 11)
        else:
            eye_dh = interesection_3P(landmark_left_eye, 0, 8, 12)

        eye_height = eye_uh + eye_dh
        eye_left_width = distance(landmark_left_eye, 0, 8)
        eye_aspect_ratio = round(eye_left_width / eye_height, 2)  # 종횡비
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
        e_ratio = round(eye_left_width / A1, 2)

        if eye_aspect_ratio <= 2.5:
            eye_shape["eye_shape"] = 0
        elif 2.5 < eye_aspect_ratio <= 2.75:
            eye_shape["eye_shape"] = 1
        elif 2.75 < eye_aspect_ratio <= 3.3:
            eye_shape["eye_shape"] = 2
        elif 3.3 < eye_aspect_ratio <= 3.5:
            eye_shape["eye_shape"] = 3
        elif 3.5 < eye_aspect_ratio:
            eye_shape["eye_shape"] = 4
        else:
            eye_shape["eye_shape"] = None

        a_angle = round(angle3(landmark_left_eye, 4, 0, 8), 2)
        b_angle = round(angle3(landmark_left_eye, 8, 0, 12), 2)
        angle_ratio = round(a_angle / b_angle, 2)
        eye_lin = round(lin(landmark_left_eye, 8, 0), 2)
        if (eye_lin <= 0.03) & (angle_ratio >= 1.8):
            eye_shape["eye_lin"] = 0
        elif 0.03 < eye_lin < 0.17:
            eye_shape["eye_lin"] = 1
        elif 0.17 <= eye_lin:
            eye_shape["eye_lin"] = 2
        else:
            eye_shape["eye_lin"] = None

        eye_d_lst = [landmark_left_eye[8], landmark_right_eye[8]]
        eye_distance = distance(eye_d_lst, 0, 1)
        d_ratio = round(d3 / A1, 2)

        if (eye_distance >= 130) & (d_ratio >= 0.29):
            eye_shape["eye_distance"] = 0
        elif (100 > eye_distance) & (0.22 >= d_ratio):
            eye_shape["eye_distance"] = 2
        else:
            eye_shape["eye_distance"] = 1

        eye_shape_labels = eye_shape_labels[int(eye_shape["eye_shape"])]
        eye_shape["eye_shape"] = eye_shape_labels
        eye_lin_labels = eye_lin_labels[int(eye_shape["eye_lin"])]
        eye_shape["eye_lin"] = eye_lin_labels
        eye_distance_labels = eye_distance_labels[int(eye_shape["eye_distance"])]
        eye_shape["eye_distance"] = eye_distance_labels
        eyeshape_result = {"label": [eye_shape["eye_shape"], eye_shape["eye_lin"], eye_shape["eye_distance"]]}

        # 눈썹 분류
        EBD = []
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

        el0 = round(eyebrowlin(landmark_left_eyebrow, 0, 1), 3)
        el1 = round(eyebrowlin(landmark_left_eyebrow, 1, 2), 3)
        el2 = round(eyebrowlin(landmark_left_eyebrow, 2, 3), 3)
        el3 = round(eyebrowlin(landmark_left_eyebrow, 3, 4), 3)

        if el0 == 0:
            el0 = 0.1
        if el1 == 0:
            el1 = 0.1
        if el2 == 0:
            el2 = 0.1
        if el3 == 0:
            el3 = 0.1

        lm0 = abs(round((el1 - el0) / el0 * 100))
        lm1 = abs(round((el2 - el1) / el1 * 100))
        lm2 = abs(round((el3 - el2) / el2 * 100))

        result = [EBD[1], EBD[2], EBD[3], EBD[4], EBD[5],
                  el0, el1, el2, el3, lm0, lm1, lm2]
        re = [eyebrow_min_max_normalize(result)]
        y_predict = EyebrowShapeModel.predict(np.array(re))
        label = eyebrow_labels[int(y_predict[0])]
        y_predict = EyebrowShapeModel.predict_proba(np.array(re))
        confidence = y_predict[0][y_predict[0].argmax()]
        for i in range(len(y_predict[0])):
            y_predict[0][i] = round(y_predict[0][i], 2) * 100
        eyebrowshape_result = {"label": label, "confidence": round((confidence * 100), 2)}
        result = [eyeshape_result, eyebrowshape_result]
        return result
class NoseShape:
    def __init__(self):
        super(NoseShape).__init__()
    def noseshape(landmark_dict):
        nose_shape = {}
        face_direction_lst = landmark_dict["noseshape"][0]
        landmark_nose = landmark_dict["noseshape"][1]
        # labels = ['big', 'small', 'long', 'short', 'normal']
        labels = ['큰 코', '작은 코', '긴 코', '짧은 코', '보통']


        face_height = distance(face_direction_lst, 3, 4)
        face_width = distance(face_direction_lst, 5, 6)

        nose_height = distance(landmark_nose, 0, 1)
        nose_width = distance(landmark_nose, 2, 3)
        nose_ratio = round(nose_width / nose_height, 2)

        nose_height_ratio = round(nose_height / face_height, 2)
        nose_width_ratio = round(nose_width / face_width, 2)

        if (nose_height >= 160) & (nose_width >= 139):
            result = labels[0]
        elif (nose_height <= 145) & (nose_width <= 120):
            result = labels[1]
        elif nose_height >= 170:
            result = labels[2]
        elif nose_height <= 150:
            result = labels[3]
        else:
            result = labels[4]

        nose_shape = {"label" : result}
        return nose_shape
class MouthShape:
    def __init__(self):
        super(MouthShape).__init__()
    def mouthshape(landmark_dict):
        mouth_shape = {}
        face_direction_lst = landmark_dict[ "mouthshape"][0]
        landmark_mouth = landmark_dict[ "mouthshape"][1]
        landmark_nose = landmark_dict[ "mouthshape"][2]
        # labels = ['big','small','upper thick', 'lower thick', 'thick', 'thin', 'normal']
        labels = ['큰 입','작은 입','윗 입술이 두꺼운 입', '아랫 입술이 두꺼운 입', '입술이 두꺼운 입', '입술이 가는 입', '보통']

        # index 7 x 좌표 index 6 x 좌표로 수정
        landmark_mouth[7] = (landmark_mouth[6][0], landmark_mouth[7][1])
        philtrum = []
        philtrum.append(face_direction_lst[0])
        philtrum.append(landmark_mouth[0])
        # 턱 너비
        chin = distance(face_direction_lst, 1, 2)
        # 인중 거리
        # philtrum_distance = distance(philtrum, 0, 1)
        # 코 너비
        nose_width = distance(landmark_nose, 2, 3)
        # 입 너비
        mouth_width = distance(landmark_mouth, 4, 5)
        upper_mouth_height = distance(landmark_mouth, 6, 7)
        lower_mouth_height = distance(landmark_mouth, 2, 3)
        mouth_height = upper_mouth_height + lower_mouth_height

        # 입꼬리 기울기
        mouth_tail = round(mouth_lin(landmark_mouth, 9, 5), 2)
        if mouth_tail >= - 0.1:
            tail_result = '입 꼬리가 올라간'
        else:
            tail_result = '입 꼬리가 내려간'
        nose_mouth_ratio = round(mouth_width / nose_width, 2)
        # mouth_ratio = round(mouth_width / mouth_height, 2)
        chin_mouth_ratio = round(mouth_width / chin, 2)

        lip_ratio = round((lower_mouth_height - upper_mouth_height) / upper_mouth_height * 100, 2)

        if (nose_mouth_ratio >= 1.45) & (chin_mouth_ratio >= 0.45):
            result = labels[0]
        elif (nose_mouth_ratio <= 1.2) & (chin_mouth_ratio <= 0.33):
            result = labels[1]
        elif (upper_mouth_height > lower_mouth_height) & (lip_ratio < -5) & (upper_mouth_height > 30):
            result = labels[2]
        elif (upper_mouth_height < lower_mouth_height) & (lip_ratio > 40) & (lower_mouth_height > 30):
            result = labels[3]
        elif (upper_mouth_height >= 30) & (lower_mouth_height >= 35):
            result = labels[4]
        elif (upper_mouth_height <= 26) & (lower_mouth_height <= 26):
            result = labels[5]
        else:
            result = labels[6]

        mouth_shape = {"label" : [result, tail_result]}
        return mouth_shape