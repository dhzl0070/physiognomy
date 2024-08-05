import collections
import math

# 분산 평균
# mean_std = [('curvature_radius', 235.0672868542629, 37.363630337204306),
#             ('chin_angle_inclease', 33.95233095338093, 3.869353715878673),
#             ('chin_gradient', 56.21965560688786, 7.862214329379974),
#             ('chin_born_angle', 157.27362452750944, 3.4301840762970204),
#             ('chin_shape_angle', 91.5787484250315, 6.606872977900956),
#             ('brow_shape_angle', 73.13607727845444, 3.0186951408747538),
#             ('ratio_width_height', 1.3771146577068458, 0.0669422064781791),
#             ('raito_brow_chin', 0.8714006719865602, 0.020992576425563538),
#             ('ratio_width_chin', 0.8792377152456952, 0.02162902110685348)]
mean_std = [('curvature_radius', 237.75056603773587, 32.2887688181104),
            ('chin_angle_inclease', 34.06775300171527, 3.7987706227553635),
            ('chin_gradient', 56.09069468267582, 7.791183335530284),
            ('chin_born_angle', 157.55102915951971, 2.5807646700481217),
            ('chin_shape_angle', 91.63250428816467, 6.814531950258692),
            ('brow_shape_angle', 73.39558319039452, 3.477295538975342),
            ('ratio_width_height', 1.3731432246998283, 0.07468303566417477),
            ('raito_brow_chin', 0.8712971698113208, 0.020888559793352475),
            ('ratio_width_chin', 0.8794018010291594, 0.022342602818414114)]

# 정규화
def min_max_normalize(lst):
    normalized = []
    for i in range(len(lst)):
        normalized_num = (lst[i] - mean_std[i][1]) / mean_std[i][2]
        normalized.append(normalized_num)
    return normalized

# collections.nametuple을 이용하여 튜플을 리스트로 하나씩 담아주는 함수
Point2D = collections.namedtuple('Point2D', ['x', 'y'])
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

# 사각턱 계산을 위한 교차점 좌표 추출
def square_interesection(landmark_tuple, int1, int2, int3, int4):
    P1 = Point2D(landmark_tuple[int1][0], landmark_tuple[int1][1])
    P2 = Point2D(landmark_tuple[int2][0], landmark_tuple[int2][1])
    P3 = Point2D(landmark_tuple[int3][0], landmark_tuple[int3][1])
    P4 = Point2D(landmark_tuple[int4][0], landmark_tuple[int4][1])
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

