import os
import random
import shutil


def copyFile(imagedir, labeldir):
    # 1
    pathDir = os.listdir(imagedir)

    # 2
    sample = random.sample(pathDir, 200)
    print(sample)

    # 3
    for name in sample:
        shutil.copyfile(imagedir + name, imgtarDir + name)
        shutil.copyfile(labeldir + name, labtardir + name)
    print("copy finish")

    for name in sample:
        os.remove(os.path.join(imagedir, name))
        os.remove(os.path.join(labeldir, name))
    print("delete finish")

if __name__ == '__main__':
    imagedir = r'/home/mel/2020_comp/Brain_Mri/datasets/0717 Release of training data/image//'
    imgtarDir = r'/home/mel/2020_comp/Brain_Mri/datasets/validation/image//'
    labeldir = r'/home/mel/2020_comp/Brain_Mri/datasets/0717 Release of training data/label//'
    labtardir = r'/home/mel/2020_comp/Brain_Mri/datasets/validation/label//'
    copyFile(imagedir, labeldir)
