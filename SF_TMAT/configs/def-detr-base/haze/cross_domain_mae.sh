BATCH_SIZE=8
DATA_ROOT="/data2/uittogether/LuuTru/huytnc/dataset/uit_drone"
OUTPUT_DIR=./outputs/uitdrone2foggy/cross_domain_mae

CUDA_VISIBLE_DEVICES=3 python -u main.py \
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
--epoch 40 \
--epoch_lr_drop 30 \
--mode cross_domain_mae \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/../source_only/model_best.pth \
