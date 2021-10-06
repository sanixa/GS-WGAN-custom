# +
import os, shutil

shutil.rmtree('data/celeba/img_align_celeba')
os.makedirs('data/celeba/img_align_celeba', exist_ok=True)

path = os.path.join('/work/u5366584/exp', 'datasets', 'celeba/celeba/img_align_celeba/')
dirs = os.listdir( path )

with open('source/index_20k.txt') as f:
    lines = f.readlines()
    
for i in range(len(lines)):
    lines[i] = int(lines[i][:-1])
    
#import ipdb;ipdb.set_trace()
for file in dirs:
    if int(file.split('.')[0]) in lines:
        shutil.copyfile(path + file, 'data/celeba/img_align_celeba/' + file)
