import os
import json

i, nimgs = 0, 13
subset_file = f"subsets/imagenet_subsets1/{nimgs}imgs_class.txt"
tokencut_dir = f"subsets/tokencut_subsets1/{nimgs}imgs_class"
in_classes = "imagenet_classes.json"

with open(in_classes, 'r') as file:
    classes_dict = json.load(file)

tc_pths = os.listdir(tokencut_dir)
missing_idx, missing_files, missing_cls = [], [], []

with open(subset_file, 'r') as rfile:
    for file in rfile.readlines():
        in_filename = file.split('.')[0]
        in_class = classes_dict[file.split('_')[0]]
        tc_filename = in_filename + "_tokencut_bfs.pt"
        if tc_filename not in tc_pths:
            missing_files.append(in_filename)
            missing_idx.append(i)
            missing_cls.append(in_class)
        i += 1

print(f"\nNumber of missing files is {len(missing_files)}")
print(f"\nMissing indices in {nimgs}imgs_class subset: {missing_idx}")
print(f"\nMissing classes in {nimgs}imgs_class subset: {missing_cls}")
print(f"\nMissing files in {nimgs}imgs_class subset: {missing_files}\n")