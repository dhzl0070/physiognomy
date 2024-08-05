import pandas as pd
df = pd.read_excel(r'D:\workplace\test\shape\ex\train2\model\faceshape1.2.xlsx',header=None, names=[0, 1, 2], engine="openpyxl")
# 관상 뜻풀이 기본 틀
face = {
                "하트형 얼굴": df.loc[0][2],
                "계란형 얼굴": df.loc[1][2],
                "형태인 원형 얼굴": df.loc[2][2],
                "사각형 얼굴": df.loc[3][2],
                "마름모형 얼굴": df.loc[4][2],
                "타원형 얼굴": df.loc[5][2],
    }
eye = {
                "동그란 눈": df.loc[6][2],
                "큰 눈": df.loc[7][2],
                "보통 눈": df.loc[8][2],
                "작은 눈": df.loc[9][2],
                "가는 눈": df.loc[10][2],
        }
eyetail = {
                "눈 끝이 위로 올라간": df.loc[11][2],
                "눈 끝이 일자인": df.loc[12][2],
                "눈 끝이 아래로 쳐진": df.loc[13][2],
}
eyedistance = {
                "눈 사이가 넓은": df.loc[14][2],
                "눈 사이가 보통": df.loc[15][2],
                "눈 사이가 좁은": df.loc[16][2],
}
eyebrow = {
                "올라간 눈썹": df.loc[17][2],
                "일자 눈썹": df.loc[18][2],
                "아치형 눈썹": df.loc[19][2],
}
nose = {
                "큰 코" : df.loc[20][2],
                "작은 코": df.loc[21][2],
                "보통": df.loc[22][2],
                "긴 코": df.loc[23][2],
                "짧은 코": df.loc[24][2],
}
mouth = {
                "큰 입" : df.loc[25][2],
                "작은 입": df.loc[26][2],
                "보통": df.loc[27][2],
                "입술이 두꺼운 입": df.loc[28][2],
                "입술이 가는 입": df.loc[29][2],
                "윗 입술이 두꺼운 입": df.loc[30][2],
                "아랫 입술이 두꺼운 입": df.loc[31][2],
}
mouthtail = {
                "입 꼬리가 올라간": df.loc[32][2],
                "입 꼬리가 내려간": df.loc[34][2],
}
