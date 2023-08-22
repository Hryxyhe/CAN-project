listTest=(
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_221.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_222.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_223.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_224.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_225.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_226.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_227.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_228.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_229.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_230.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_231.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_232.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_233.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_234.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_235.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_236.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_237.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_238.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_239.pth
    checkpoints/CAN_2022-09-17-16-04_decoder-AttDecoder/CAN_2022-09-17-16-04_decoder-AttDecoder_WordRate-0.0000_ExpRate-0.0000_240.pth
)

for file_path in ${listTest[@]}
do
CUDA_VISIBLE_DEVICES=5 python inference.py --dataset CROHME --model_path $file_path --image_path datasets/CROHME/19_test_images.pkl --label_path datasets/CROHME/19_test_labels.txt
done