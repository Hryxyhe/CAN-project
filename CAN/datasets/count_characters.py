train_labels_file = "/home/ipad_ocr/CAN/datasets/HME100K/train_labels.txt"
test_labels_file = "/home/ipad_ocr/CAN/datasets/HME100K/test_labels.txt"


all_items = set()

# 读取训练集标签文件并记录字符
with open(train_labels_file, "r") as file:
    for line in file:
        items = line.strip().split()  # 使用空格划分字符和词语
        items = [item for item in items if not item.endswith(".jpg")]  # 去掉以".jpg"为后缀的词语
        all_items.update(items)
        print(f"已记录项个数：{len(all_items)}", end="\r")

# 读取测试集标签文件并记录字符
with open(test_labels_file, "r") as file:
    for line in file:
        items = line.strip().split()  # 使用空格划分字符和词语
        items = [item for item in items if not item.endswith(".jpg")]  # 去掉以".jpg"为后缀的词语
        all_items.update(items)
        print(f"已记录项个数：{len(all_items)}", end="\r")

with open("/home/ipad_ocr/CAN/datasets/HME100K/word_dict.txt", "w") as file:
    for char in sorted(all_items):
        file.write(char + "\n")

print("字符字典生成完成！")