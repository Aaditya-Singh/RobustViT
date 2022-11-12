SS=subsets1
NIMGS=5

python tokencut_generate_segmentation.py \
  --image-folder /srv/datasets/ImageNet/imagenet/train/ \
  --subset-path subsets/imagenet_${SS}/${NIMGS}imgs_class.txt \
  --out-dir subsets/tokencut_${SS}/${NIMGS}imgs_class/