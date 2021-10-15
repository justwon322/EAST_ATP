# -*- coding: utf-8 -*-
import argparse
import glob
import os

from PIL import Image, ImageDraw
import json

'''
잘린이미지 좌표만 변경하시면 됩니다.
이미지 변환 끝나면
recognition/create_lmdb_dataset.py
수행하시면 됩니다. 근데 데이터 이미지가 워낙많아서 테스트용으로 쓰실거면 원본 이미지폴더에 이미지 1개만 남기시는거 추천합니다... 순식간에 몇만개가 생겨버려요
'''
def parsing(args):
    gt_path = glob.glob(os.path.join(args.train_gt_path, "**", "**", "**", "*.json"))  # 파일경로


    text = [] # 어노테이션 txt파일 저장용

    for i, gt in enumerate(gt_path):
        with open(gt, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            annotations = json_data["annotations"] # text / boundig box 접근
            img = gt[:-4] + "JPG"
            img = img.replace("json", "jpg")
            image1 = Image.open(img)
            for j, object in enumerate(annotations):
                gt_text = object['annotation.text']

                x, y, w, h = object['annotation.bbox']
                croppedImage = image1.crop((x, y, x + w, y + h)) # 이미지 자르기 (사각형 왼쪽위 꼭지점 좌표 , 오른쪽 아래 꼭지점좌표)
                croppedImage.save('./data/training/'+str(i) + '_' + str(j) + '.jpg') #이미지 저장 위치 변경 필요
                t = str(i) + '_' + str(j) + '.jpg\t' + gt_text+'\n'
                text.append(t)


    f = open('./data/train_gt.txt', 'w', encoding='utf-8') #gt.txt파일 저장 위치 변경 필요
    f.writelines(text)
    f.close()


def main():
    # parser
    parser = argparse.ArgumentParser(description="---#---")

    parser.add_argument('--train_gt_path', type=str, default="../detection/dataset/json") # 파일경로


    args = parser.parse_args()
    parsing(args)


if __name__ == '__main__':
    main()
