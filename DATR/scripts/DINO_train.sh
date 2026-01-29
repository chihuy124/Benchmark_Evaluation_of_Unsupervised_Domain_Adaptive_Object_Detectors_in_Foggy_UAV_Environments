export CUDA_VISIBLE_DEVICES=2 && python main.py \
--dataset_file hazy \
--output_dir logs/hazydet/DINO_hazydet2foggy/R50_ms4 \
-c config/DA/hazydet2foggy/DINO_4scale_hazydet2F.py \
--options dn_scalar=100 embed_init_tgt=TRUE \
dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
dn_box_noise_scale=1.0
