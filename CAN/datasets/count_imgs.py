import os

def count_files_with_extension(folder_path, extension):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(extension):
            count += 1
    return count

image_folder_path = '/home/ipad_ocr/CAN/datasets/train_images'
image_extension = '.jpg'

total_image_count = count_files_with_extension(image_folder_path, image_extension)
print(f"Total number of {image_extension} files: {total_image_count}")
