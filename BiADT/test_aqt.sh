# used for inference on c2fc, with AQT
batch_size=2

coco_path=/data2/uittogether/LuuTru/huytnc/dataset
output_dir=biADT8_r50_uitdrone_test_aqt
epochs=50
lr_drop=40
num_queries=300
backbone=resnet50
resume_weights=biADT8_r50_uitdrone/checkpoint0044_beforedrop.pth
dataset_file=uitdrone
with_aqt=1
space_q=0.10
chann_q=0.001
insta_q=0.001

python -u main.py -m dab_deformable_detr \
        --output_dir ${output_dir}      \
        --backbone ${backbone}          \
        --batch_size ${batch_size}      \
        --lr 5e-5                       \
        --lr_drop ${lr_drop}            \
        --num_queries ${num_queries}    \
        --epochs $epochs                \
        --coco_path ${coco_path}        \
        --hidden_dim 384                \
        --resume ${resume_weights}      \
        --with_aqt ${with_aqt}          \
        --space_q ${space_q}            \
        --chann_q ${chann_q}            \
        --insta_q ${insta_q}            \
        --eval                          \
        --dataset_file ${dataset_file}  