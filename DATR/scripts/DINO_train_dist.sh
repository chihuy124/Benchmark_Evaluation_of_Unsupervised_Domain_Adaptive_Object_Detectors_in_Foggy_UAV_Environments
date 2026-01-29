export CUDA_VISIBLE_DEVICES=5,6 && python -m torch.distributed.launch --master_port=29603 --nproc_per_node=2 --use_env main.py \
--dataset_file city \
--output_dir logs/DINO_uitdrone2foggy/R50_ms4 \
-c config/DA/uitdrone2foggy/DINO_4scale_uitdrone2F.py \
--options dn_scalar=100 embed_init_tgt=TRUE \
dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
dn_box_noise_scale=1.0


