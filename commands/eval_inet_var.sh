# TODO: hardcoded data path
SS=subsets1
NIMGS=1
DATA=IN1k
MODEL=MSN_VITB

python imagenet_eval_robustness.py \
  --data /srv/datasets/ImageNet/imagenet/val/ \
  --evaluate --batch-size 256 \
  --checkpoint logs/${MODEL}/IN_${SS}_${NIMGS}imgs_class/model_best.pth.tar

