import cv2
import glob
import os
import itertools


# if os.path.exists(absimg_dir) == False: # 저장 경로 없으면 만들기
#    os.makedirs(absimg_dir)
# 폴더안에 있는 텍스트 리스트 불러와 (x,y,w,h)좌표로 나오게 한 후 loc 리스트에 저장
def convertCoordination():
    my_loc = []
    len_loc = []
    txt_path = './utils/temp'#
    txt_dir = glob.glob(txt_path + '/*.txt')
    img_path = './output'
    img_dir = glob.glob(img_path + '/*.jpg')
    for _img, _txt in zip(img_dir, txt_dir):
        # 빈 리스트 생성
        coordinate = []
        # dh dw : 이미지의 가로, 세로 크기
        # _ : 채널(RGB = 3)
        img = cv2.imread(_img)
        dh, dw, _ = img.shape
        # YOLO 좌표가 있는 텍스트 파일 불러옴
        fl = open(_txt, 'r')
        data = [line[:-1] for line in fl.readlines()]
        fl.close()
        for dt in data:
            # Split string to float
            x1, y1, x2, y2, x3, y3, x4, y4, score = map(float, dt.split())
            # l : left , r : right , t : top, b : bottom
            # x = l , y = t , w = r-l, h = b-t
            x = int(x1)
            y = int(y2)
            w = int(x2 - x1)
            h = int(y3 - y2)

            # l = int((x - w / 2)) #63.94
            # r = int((x + w / 2)) #101.94
            # t = int((y - h / 2))
            # b = int((y + h / 2))
            # # 좌표 후처리(없어도 무관)
            # # left 값이 - 일 경우 0으로 저장
            # if l < 0:
            #     l = 0
            # # right 값이 (전체 이미지 사이즈) - 1 보다 클 경우
            # if r > dw - 1:
            #     r = dw - 1
            # # top 값이 - 일 경우 0으로 저장
            # if t < 0:
            #     t = 0
            # # bottom 값이 (전체 이미지 사이즈) - 1 보다 클 경우
            # if b > dh - 1:
            #     b = dh - 1
            # # 리스트에 저장
            coordinate.append([x, y, w, h])
        coordinate.sort(key=lambda x: x[1])

        s = []
        loc = []
        for i in range(1, len(coordinate)):
            if abs(coordinate[i - 1][1] - coordinate[i][1]) < 10:
                s.append(coordinate[i - 1])
            else:
                if len(s) == 0:
                    loc.append([coordinate[i - 1]])
                else:
                    s.append(coordinate[i - 1])
                    s.sort(key=lambda x: x[0])
                    loc.append(s)
                    s = []

        s.append(coordinate[i - 1])
        s.sort(key=lambda x: x[0])
        loc.append(s)

        len_loc.append(list(map(lambda x: len(x), loc)))
        my_loc.append(list(itertools.chain(*loc)))

    return len_loc, my_loc


# 만들어낸 loc 리스트를 불러와 좌표값에 따른 이미지 추출하여 저장
def read_img_by_coord(loc):
    img_list = []
    img_path = './output'
    img_dir = glob.glob(img_path + '/*.jpg')
    for _img, _loc in zip(img_dir, loc):
        print("make image", os.path.basename(_img[:-4]))
        org_image = cv2.imread(_img, cv2.IMREAD_GRAYSCALE)
        cv_list = []
        for i in _loc:
            img_trim = org_image[i[1]:i[1] + i[3], i[0]:i[0] + i[2]]  # trim한 결과를 img_trim에 담는다
            cv_list.append(img_trim)
        img_list.append(cv_list)
    return img_list

# location = getcoord()
# trim_image = read_img_by_coord(location) #trim_image 변수에 결과물을 넣는다
