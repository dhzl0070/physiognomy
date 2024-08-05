from face_shape_classify.main_func import *
from face_shape_classify.utils.lucky_label import *
import time
import warnings
import sys
warnings.filterwarnings(action='ignore')

# save_check = True
save_check = False

img = r"D:\workplace\test\oval (927).jpg"
img_name = img.split(".")[0].split("\\")[-1]
result_dict = {}
try:
    start = time.time()  # 시작 시간 저장
    img = margin(iris_rotate(img))
    print("rotate time :", round(time.time() - start, 3))  # 현재시각 - 시작시간 = 실행 시간
    landmarks_total_dict = Landmarks.landmarks(img)
    if landmarks_total_dict != None:
        total_face_shape, classifier_lst = FaceShape.faceshape(landmarks_total_dict)
        face_shape = total_face_shape['label']
        face_shape_lst = f"{face_shape}({classifier_lst})"
        eye_total = EyeShape.eyeshape(landmarks_total_dict)[0]
        eye_shape = eye_total['label'][0]
        eye_tail_shape = eye_total['label'][1]
        eye_distance = eye_total['label'][2]
        eye_brow_shape = EyeShape.eyeshape(landmarks_total_dict)[1]['label']
        nose_shape = NoseShape.noseshape(landmarks_total_dict)['label']
        mouth_total = MouthShape.mouthshape(landmarks_total_dict)
        mouth_shape = mouth_total['label'][0]
        mouth_tail_shape = mouth_total['label'][1]

        # result_dict = {"results": [
        #     {"title": "얼굴형",
        #      "label": face_shape,
        #      "content": face[face_shape]},
        #     {"title": "눈 모양",
        #      "label": eye_shape,
        #      "content": eye[eye_shape]},
        #     {"title": "눈꼬리 모양",
        #      "label": eye_tail_shape,
        #      "content": eyetail[eye_tail_shape]
        #      },
        #     {"title": "눈 사이 거리",
        #      "label": eye_distance,
        #      "content": eyedistance[eye_distance]
        #      },
        #     {"title": "눈썹 모양",
        #      "label": eye_brow_shape,
        #      "content": eyebrow[eye_brow_shape]
        #      },
        #     {"title": "코 모양",
        #      "label": nose_shape,
        #      "content": nose[nose_shape]
        #      },
        #     {"title": "입 모양",
        #      "label": mouth_shape,
        #      "content": mouth[mouth_shape]
        #      },
        #     {"title": "입꼬리 모양",
        #      "label": mouth_tail_shape,
        #      "content": mouthtail[mouth_tail_shape]
        #      }
        # ]}
        result_dict = {"results": [
            {"title": "얼굴형",
             "label": face_shape},
            {"title": "눈 모양",
             "label": eye_shape},
            {"title": "눈꼬리 모양",
             "label": eye_tail_shape},
            {"title": "눈 사이 거리",
             "label": eye_distance
             },
            {"title": "눈썹 모양",
             "label": eye_brow_shape
             },
            {"title": "코 모양",
             "label": nose_shape
             },
            {"title": "입 모양",
             "label": mouth_shape
             },
            {"title": "입꼬리 모양",
             "label": mouth_tail_shape
             }
        ]}
        print(result_dict["results"])
    else:
        print("Rotate Error!!")
    print("total time :", round(time.time() - start, 3))  # 현재시각 - 시작시간 = 실행 시간
except Exception as e:
    print(e)

