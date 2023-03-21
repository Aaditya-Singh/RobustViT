SS=subsets1
NIMGS=5
START=0
END=5000

for ((INDEX=${START}; INDEX<${END}; INDEX++));
do
    echo -e "Running TokenCut for image ${INDEX} in ${SS}/${NIMGS}imgs_class subset..."
    python tokencut_generate_segmentation.py \
    --image-folder ../datasets/ImageNet/imagenet/train/ \
    --subset-path subsets/imagenet_${SS}/${NIMGS}imgs_class.txt \
    --index ${INDEX} \
    --out-dir subsets/tokencut_${SS}/${NIMGS}imgs_class/ \
    --pretrained_pth pretrained/dino/dino_vitbase16_pretrain.pth
    echo -e "Finished TokenCut for image ${INDEX} in ${SS}/${NIMGS}imgs_class subset!\n"
done