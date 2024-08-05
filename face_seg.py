from face_shape_classify.main_func import *
from ultralytics import YOLO
import time
from tqdm import tqdm
import glob, os, shutil
model = YOLO(r'C:\Users\inforex\PycharmProjects\CUDA\yolov8\3shape_seg.pt')

img_dir = r'D:\workplace\seg\test\square'
round_dir = r'D:\workplace\seg\test'
square_dir = r'D:\workplace\seg\test\square'
heart_dir = r'D:\workplace\seg\test\heart'
h_cnt = 0
r_cnt = 0
s_cnt = 0
for f in tqdm(glob.glob(os.path.join(img_dir, "*.*"))):
    img = f
    img_name = img.split(".")[0].split("\\")[-1]
    result_dict = {}
    try:
        start = time.time()  # 시작 시간 저장
        # img = margin(iris_rotate(img))
        # landmarks_total_dict = Landmarks.landmarks(img)
        # if landmarks_total_dict != None:
        results = model(img, conf=0.8, save_txt=True)  # predict on an image
        # cls = results[0].boxes.cls
        # if cls == 0:
        #     cv2.imwrite(f"{heart_dir}/{img_name}.jpg", img)
        #     h_cnt += 1
        # if cls == 1:
        #     cv2.imwrite(f"{square_dir}/{img_name}.jpg", img)
        #     s_cnt += 1
        # if cls == 2:
        #     cv2.imwrite(f"{round_dir}/{img_name}.jpg", img)
        #     r_cnt += 1
        # else:
        #     print(f)
    except Exception as e:
        # print(e)
        pass
# print(f'하트형 : {h_cnt}, 사각형 : {s_cnt}, 원형 {r_cnt}')




