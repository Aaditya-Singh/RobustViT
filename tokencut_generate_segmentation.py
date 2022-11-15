import argparse

import torch
import os
import collections

import sys
from tqdm import tqdm

sys.path.append('./TokenCut/model')
sys.path.append('./TokenCut/unsupervised_saliency_detection')
import dino # model

import object_discovery as tokencut
import argparse
import utils
import bilateral_solver
import os

from shutil import copyfile
import PIL.Image as Image
import cv2
import numpy as np

from torchvision import transforms
import metric
import matplotlib.pyplot as plt
import skimage
import warnings
warnings.filterwarnings("ignore")
# NOTE: Memory leak persists when using GPU; switch to cpu instead
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from tokencut_image_dataset import RobustnessDataset

parser = argparse.ArgumentParser(description='Generate Seg maps')

parser.add_argument('--image-folder', metavar='path', default='/srv/datasets/ImageNet/imagenet/train/',
                    help='imagenet folder path')

parser.add_argument('--subset-path', metavar='path', default='subsets/imagenet_subsets1/1imgs_class.txt',
                    help='imagenet subset path')

parser.add_argument('--index', metavar='index', type=int, default=None, 
                    help='index of image in the subset to run TokenCut on')

parser.add_argument('--out-dir', type=str, default='subsets/tokencut_subsets1/', 
                    help='output directory')

parser.add_argument('--pretrained_pth', type=str, default='pretrained/dino/dino_vitbase16_pretrain.pth',
                    help="path for pretrained model")

parser.add_argument('--vit-arch', type=str, default='base', choices=['base', 'small'], help='which architecture')

parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')

parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')

parser.add_argument('--tau', type=float, default=0.2, help='Tau for tresholding graph')

parser.add_argument('--sigma-spatial', type=float, default=16, help='sigma spatial in the bilateral solver')

parser.add_argument('--sigma-luma', type=float, default=16, help='sigma luma in the bilateral solver')

parser.add_argument('--sigma-chroma', type=float, default=8, help='sigma chroma in the bilateral solver')

parser.add_argument('--dataset', type=str, default=None, choices=['ECSSD', 'DUTS', 'DUT', None], help='which dataset?')

parser.add_argument('--nb-vis', type=int, default=100, choices=[1, 200], help='nb of visualization')

ImageItem = collections.namedtuple('ImageItem', ('image_name', 'tag'))


def mask_color_compose(org, mask, mask_color = [173, 216, 230]) :

    mask_fg = mask > 0.5
    rgb = np.copy(org)
    rgb[mask_fg] = (rgb[mask_fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)

    return Image.fromarray(rgb)

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(), \
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])


def get_tokencut_binary_map(img_pth, backbone,patch_size, tau, resize_size=None) :

    I = Image.open(img_pth).convert('RGB')
    if resize_size is not None:
        I = I.resize(resize_size)

    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I, patch_size)

    feat = backbone(ToTensor(I_resize).unsqueeze(0).to(device))[0].to('cpu')

    seed, bipartition, eigvec = tokencut.ncut(feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], \
        tau, device='cpu')
    return bipartition, eigvec


def save_single_tokencut(img_pth, backbone, args):
    with torch.no_grad():        
        bipartition, eigvec = get_tokencut_binary_map(img_pth, backbone, args.patch_size, args.tau)
        output_solver, binary_solver = bilateral_solver.bilateral_solver_output(img_pth, bipartition, \
            sigma_spatial=args.sigma_spatial, sigma_luma=args.sigma_luma, sigma_chroma=args.sigma_chroma)
        mask1 = torch.from_numpy(bipartition).to('cpu')
        mask2 = torch.from_numpy(binary_solver).to('cpu')
        if metric.IoU(mask1, mask2) < 0.5:
            binary_solver = binary_solver * -1

        # output segmented image
        img_name = img_pth.split("/")[-1]
        out_name = os.path.join(out_dir, img_name)
        out_lost = os.path.join(out_dir, img_name.replace('.JPEG', '_tokencut.JPEG'))
        out_bfs = os.path.join(out_dir, img_name.replace('.JPEG', '_tokencut_bfs.JPEG'))
        out_gt = os.path.join(out_dir, img_name.replace('.JPEG', '_gt.JPEG'))

        org = Image.open(img_pth).convert('RGB')
        # plt.imsave(fname=out_eigvec, arr=eigvec, cmap='cividis')
        # mask_color_compose(org, bipartition).save(out_lost)
        # mask_color_compose(org, binary_solver).save(out_bfs)
        # mask_color_compose(org, seg_map).save(out_gt)

        # torch.save(bipartition, os.path.join(out_dir, img_name.replace('.JPEG', '_tokencut.pt')))
        torch.save(binary_solver, os.path.join(out_dir, img_name.replace('.JPEG', '_tokencut_bfs.pt')))
    return


if __name__ == '__main__':

    args = parser.parse_args()

    feat_dim = 768
    args.patch_size = 16
    args.vit_arch = 'base'

    backbone = dino.ViTFeat(args.pretrained_pth, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
    msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
    # print(msg)
    backbone.eval()
    backbone.to(device)

    # transforms - start
    n_imgs, img_paths = 0, []
    img_folder = args.image_folder
    subset_path = args.subset_path
    out_dir = args.out_dir
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    with open(subset_path, 'r') as rfile:      
        for line in rfile:
            n_imgs += 1
            class_name, img = line.split('_')[0], line.split('\n')[0]            
            img_pth = os.path.join(img_folder, class_name, img)
            img_paths.append(img_pth)

    if args.index is not None and args.index < len(img_paths):        
        img_pth = img_paths[args.index]
        save_single_tokencut(img_pth, backbone, args)
    else:
        raise Exception(f"Index {args.index} is not in the subset file")
    exit(0)