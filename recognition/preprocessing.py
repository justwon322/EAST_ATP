#%%
import argparse
import glob
import json
import os
from tqdm import tqdm

from PIL import Image

def parsing(args):
    gt_path = glob.glob(os.path.join(args.train_gt_path, "**", "**", "**", "*.json"))

    text = []

    for i, gt in enumerate(tqdm(gt_path)):
        with open(gt, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        annotations = json_data["annotations"] 
        img = gt[:-4] + "JPG"
        img = img.replace("json", "jpg")
        if os.path.isfile(img):
            image1 = Image.open(img)
            for j, object in enumerate(annotations):
                gt_text = object['annotation.text']
                x, y, w, h = object['annotation.bbox']
                croppedImage = image1.crop((x, y, x + w, y + h))
                try:
                    croppedImage.save('./data/'+str(i) + '_' + str(j) + '.jpg')
                except:
                    continue
                t = str(i) + '_' + str(j) + '.jpg\t' + gt_text+'\n'
                text.append(t)


    f = open('./data/gt.txt', 'w', encoding='utf-8')
    f.writelines(text)
    f.close()

def main():
    parser = argparse.ArgumentParser(description="---#---")
    parser.add_argument('--train_gt_path', type=str, default="../detection/dataset/json") # 파일경로
    args = parser.parse_args()
    parsing(args)

if __name__ == '__main__':
    main()

# %%
