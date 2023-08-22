import cv2
import glob
import os
import pickle as pkl


image_path = '/home/ipad_ocr/CAN/datasets/test_images'
out_path = '/home/ipad_ocr/CAN/datasets/HME100K/test_images.pkl'
# 获取了image_path目录中所有以.jpg结尾的文件的文件路径，将这些路径存储在files列表中
files = glob.glob(os.path.join(image_path, '*.jpg'))

output = {}

for item in files:
    img = cv2.imread(item)
    # 读取图像，并使用cv2.cvtColor将图像从BGR颜色空间转换为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape

    width = int(120 * w / h)

    img = cv2.resize(img, (width, 120))
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    output[os.path.basename(item)] = binary_img
    print("Processed:", item)


with open(out_path, 'wb') as f:
    # 将整个字典写入以二进制模式打开的文件对象。这将创建一个.pkl文件
    pkl.dump(output, f)
    print("File written successfully")

