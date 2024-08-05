import collections
import math
from math import atan, sqrt, sin

# collections.nametuple을 이용하여 튜플을 리스트로 하나씩 담아주는 함수
Point2D = collections.namedtuple('Point2D', ['x', 'y'])
# 기울기
def mouth_lin(landmark_tuple, int1, int2):
    P1 = Point2D(landmark_tuple[int1][0], landmark_tuple[int1][1])
    P2 = Point2D(landmark_tuple[int2][0], landmark_tuple[int2][1])
    slope = (P2.y - P1.y) / (P2.x - P1.x)
    return -slope