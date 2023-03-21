SS=subsets1
NIMGS=1
DATA=IN1k
MODEL=MSN_VITB

python imagenet_eval_robustness.py \
  --port 1234 --model_name deit_base \
  --data ../datasets/ImageNet/imagenet/val/ --batch-size 128 \
  --checkpoint logs/${MODEL}_FT/IN_${SS}_${NIMGS}imgs_class/model_best.pth.tar