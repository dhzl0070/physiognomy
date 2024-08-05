import cv2
import os
import imageio

save_dir = r'D:\workplace\test\shape\tt'
filenames = [i for i in os.listdir(save_dir) ] ## 필터링할 때
# with imageio.get_writer(r'D:\workplace\test\shape\draw.gif', mode='I',duration=0.5) as writer:
#     for filename in filenames:
#         filename = os.path.join(save_dir , filename)
#         image = imageio.v2.imread(filename)
#         writer.append_data(image)
def draw_face(face, draw_image):
    for r in range(0, 36, 1):
        if r < 35:
            start = int(face[r][0]), int(face[r][1])
            end = int(face[r+1][0]), int(face[r+1][1])
            cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        else:
            start = int(face[35][0]), int(face[35][1])
            end = int(face[0][0]), int(face[0][1])
            cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return  draw_image
def draw_eye(left_eye, right_eye, draw_image):
    for r in range(0, 15, 1):
        if r < 14:
            start = int(left_eye[r][0]), int(left_eye[r][1])
            end = int(left_eye[r+1][0]), int(left_eye[r+1][1])
            cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        else:
            start = int(left_eye[14][0]), int(left_eye[14][1])
            end = int(left_eye[0][0]), int(left_eye[0][1])
            cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    for r in range(0, 15, 1):
        if r < 14:
            start = int(right_eye[r][0]), int(right_eye[r][1])
            end = int(right_eye[r+1][0]), int(right_eye[r+1][1])
            cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        else:
            start = int(right_eye[14][0]), int(right_eye[14][1])
            end = int(right_eye[0][0]), int(right_eye[0][1])
            cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return draw_image
def draw_eyebrow(left_eyebrow, right_eyebrow, draw_image):
    for r in range(0, 3, 1):
        start = int(left_eyebrow[r][0]), int(left_eyebrow[r][1])
        end = int(left_eyebrow[r+1][0]), int(left_eyebrow[r+1][1])
        start2 = int(right_eyebrow[r][0]), int(right_eyebrow[r][1])
        end2 = int(right_eyebrow[r+1][0]), int(right_eyebrow[r+1][1])
        cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.line(draw_image, start2, end2, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return draw_image

def draw_nose(nose, draw_image):
    for r in range(0, 7, 1):
        start = int(nose[r][0]), int(nose[r][1])
        end = int(nose[r+1][0]), int(nose[r+1][1])
        cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    for r in range(7, 27, 1):
        if r < 27:
            start = int(nose[r][0]), int(nose[r][1])
            end = int(nose[r+1][0]), int(nose[r+1][1])
            cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        else :
            start = int(nose[27][0]), int(nose[27][1])
            end = int(nose[8][0]), int(nose[8][1])
            cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return draw_image

def draw_mouth(mouth, draw_image):
    for r in range(0, 41, 1):
        if r < 40:
            start = int(mouth[r][0]), int(mouth[r][1])
            end = int(mouth[r+1][0]), int(mouth[r+1][1])
            cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        else:
            start = int(mouth[40][0]), int(mouth[40][1])
            end = int(mouth[0][0]), int(mouth[0][1])
            cv2.line(draw_image, start, end, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return draw_image