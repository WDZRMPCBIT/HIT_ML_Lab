import numpy as np
import os, sys
from PIL import Image
import torchvision.transforms as transforms
import shutil

output_dirHR = 'jpg_out'


def get_size(file):
    # 获取文件大小:KB
    size = os.path.getsize(file)
    return size / 1024


def get_outfile(infile, outfile):
    return infile


def compress_image(infile, outfile='', mb=150, step=10, quality=80):
    """不改变图片尺寸压缩到指定大小
    :param infile: 压缩源文件
    :param outfile: 压缩文件保存地址
    :param mb: 压缩目标，KB
    :param step: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件地址，压缩文件大小
    """
    o_size = get_size(infile)
    if o_size <= mb:
        return infile
    outfile = get_outfile(infile, outfile)
    while o_size > mb:
        im = Image.open(infile)
        im.save(outfile, quality=quality)
        if quality - step < 0:
            break
        quality -= step
        o_size = get_size(outfile)
    return outfile, get_size(outfile)


def resize_image(infile, outfile='', x_s=30, y_s=30):
    """修改图片尺寸
    :param infile: 图片源文件
    :param outfile: 重设尺寸文件保存地址
    :param x_s: 设置的宽度
    :return:
    """
    im = Image.open(infile)
    out = im.resize((x_s, y_s), Image.ANTIALIAS)
    outfile = get_outfile(infile, outfile)
    out.save(outfile)


# bmp 转换为jpg
def bmpToJpg(file_path):
    for fileName in os.listdir(file_path):
        newFileName = fileName[0:fileName.find("_")] + ".jpg"
        print(newFileName)
        im = Image.open(file_path + "\\" + fileName)
        im.save(file_path + "\\" + newFileName)


def image_RGB2Gray(image_path):
    # 图片路径，相对路径
    # 读取图片
    image = Image.open(image_path)
    # RGB转换我灰度图像
    image_transforms = transforms.Compose([transforms.Grayscale(1)])
    image = image_transforms(image)
    image.save(image_path)


if __name__ == "__main__":
    dirs = os.listdir('.')
    for file_name in dirs:
        if file_name.split('.')[1] != 'py':
            compress_image(file_name)
            resize_image(file_name)
            image_RGB2Gray(file_name)
