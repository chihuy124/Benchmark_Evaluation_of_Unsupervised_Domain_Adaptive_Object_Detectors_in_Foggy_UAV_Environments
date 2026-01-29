BATCH_SIZE=4
DATA_ROOT="/data2/uittogether/LuuTru/huytnc/dataset/uit_drone"
OUTPUT_DIR=./outputs/uitdrone2foggy/teaching

CUDA_VISIBLE_DEVICES=4 python -u main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 4 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset uitdrone \
--target_dataset foggy_uitdrone \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--epoch 80 \
--epoch_lr_drop 80 \
--mode teaching \
--output_dir ${OUTPUT_DIR} \
--resume "/data2/uittogether/LuuTru/huytnc/SF_TMAT/outputs/uitdrone2foggy/cross_domain_mae/model_best.pth" \
--epoch_retrain 40 \
--epoch_mae_decay 10 \
--threshold 0.3 \
--max_dt 0.45
