SS=subsets1
NIMGS=1
MODEL=MSN_VITB

python imagenet_finetune_tokencut.py \
  --data /srv/datasets/ImageNet/imagenet/train/ \
  --seg_data subsets/tokencut_${SS}/${NIMGS}imgs_class/ \
  --val_data /srv/datasets/ImageNet/imagenet/val/ \
  --experiment_folder logs/${MODEL}/IN_${SS}_${NIMGS}imgs_class/ \
  --model-name deit_base --pretrained pretrained/msn/vitb16_600ep.pth.tar \
  --epochs 50 --batch-size 8 --lr 8e-7 --temperature 0.65 \
  --lambda_acc 0.2 --lambda_background 2 --lambda_seg 0.8 --lambda_foreground 0.3 

