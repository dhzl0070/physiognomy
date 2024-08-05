import collections
import math
from math import atan, sqrt, sin

# collections.nametuple을 이용하여 튜플을 리스트로 하나씩 담아주는 함수
Point2D = collections.namedtuple('Point2D', ['x', 'y'])
def interesection_3P(tuple, int1, int2, int3):
    P1 = Point2D(tuple[int1][0], tuple[int1][1])
    P2 = Point2D(tuple[int2][0], tuple[int2][1])
    P3 = Point2D(tuple[int3][0], tuple[int3][1])
    x1, y1 = P1.x, P1.y
    x2, y2 = P2.x, P2.y
    x3, y3 = P3.x, P3.y
    m = (y2 - y1) / (x2 - x1)
    n = (y3 - y1) / (x3 - x1)
    theta = atan(m) - atan(n)
    h = sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    d = h * sin(theta)
    return int(abs(d))

# 두 선의 교차점 좌표를 구하는 함수
def interesection2(landmark_tuple, int1, int2, int3):
    P1 = Point2D((landmark_tuple[int1][0]), (landmark_tuple[int1][1]))
    P2 = Point2D((landmark_tuple[int2][0]), (landmark_tuple[int2][1]))
    P3 = Point2D((landmark_tuple[int3][0]), (landmark_tuple[int3][1]))
    P4 = Point2D((landmark_tuple[int3][0]), 640)
    x1, y1 = P1.x, P1.y
    x2, y2 = P2.x, P2.y
    x3, y3 = P3.x, P3.y
    x4, y4 = P4.x, P4.y
    px = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    py = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    p = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    x = px / p
    y = py / p
    return x, y


# 기울기
def eyebrowlin(landmark_tuple, int1, int2):
    P1 = Point2D(landmark_tuple[int1][0], landmark_tuple[int1][1])
    P2 = Point2D(landmark_tuple[int2][0], landmark_tuple[int2][1])
    slope = - ((P2.y - P1.y) / (P2.x - P1.x))
    return slope

# 곡률반경
def r_interesection(landmark_tuple, int1, int2, int3):
    P1 = Point2D(landmark_tuple[int1][0], landmark_tuple[int1][1])
    P2 = Point2D(landmark_tuple[int2][0], landmark_tuple[int2][1])
    P3 = Point2D(landmark_tuple[int3][0], landmark_tuple[int3][1])
    x1, y1 = P1.x, P1.y
    x2, y2 = P2.x, P2.y
    x3, y3 = P3.x, P3.y
    d1 = (x2 - x1) / (y2 - y1)
    d2 = (x3 - x2) / (y3 - y2)
    cx = round(((y3 - y1) + (x2 + x3) * d2 - (x1 + x2) * d1) / (2 * (d2 - d1)))
    cy = round(-d1 * (cx - (x1 + x2) / 2) + (y1 + y2) / 2)
    r = math.sqrt((math.pow((int(x1) - cx), 2) + math.pow((int(y1) - cy), 2)))
    return round(r, 2)

# 분산 평균
mean_std = [('ef0', 19.514051015996543, 8.023145111238895),
            ('ef1', 27.431412513124574, 6.470418825882711),
            ('ef2', 30.053424742140695, 4.626837013786982),
            ('ef3', 20.1437835834723, 2.7213368501931026),
            ('ef4', 0.1598418874683466, 1.1482728837169789),
            ('rad0', 0.16396825396825399, 0.0475556162462416),
            ('rad1', 0.06326910011734915, 0.06119659260732983),
            ('rad2', -0.3156024952133902, 0.07555844739970495),
            ('rad3', -0.991111111111111, 0.18117891425121477),
            ('rad_ratio0', 68.71965906985362, 30.38711036925113),
            ('rad_ratio1', 1146.995800135878, 1134.4209891617104),
            ('rad_ratio2', 234.45747637576432, 185.2625581640994)]
# 정규화
def eyebrow_min_max_normalize(lst):
    normalized = []
    for i in range(len(lst)):
        normalized_num = (lst[i] - mean_std[i][1]) / mean_std[i][2]
        normalized.append(normalized_num)
    return normalized