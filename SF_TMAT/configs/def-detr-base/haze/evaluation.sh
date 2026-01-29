BATCH_SIZE=2
DATA_ROOT="/data2/uittogether/LuuTru/huytnc/dataset/uit_drone"
OUTPUT_DIR=./outputs/def-detr-base/uitdrone2foggy/evaluation

CUDA_VISIBLE_DEVICES=2 python -u main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 4 \
--data_root ${DATA_ROOT} \
--source_dataset uitdrone \
--target_dataset foggy_uitdrone \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume "/data2/uittogether/LuuTru/huytnc/SF_TMAT/outputs/uitdrone2foggy/teaching/model_best.pth" \
