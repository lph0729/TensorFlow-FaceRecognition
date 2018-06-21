#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-6-13 下午10:44
@email: lph0729@163.com

"""
import os
import multiprocessing

from face_lib import my_api

"""生成对齐的人脸识别图像"""

input_dir = "./temp/lfw/lfw"
out_dir = "./train_faces"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if __name__ == '__main__':
    get_faces = my_api.GetAlignedFace(input_dir, out_dir)
    print(len(get_faces.pic_names))
    pic_list = get_faces.pic_names

    pool = multiprocessing.Pool(processes=4)

    pic_num = 0
    for file_name in pic_list:
        pool.apply_async(get_faces.read_photo, (file_name, pic_num))
        pic_num += 1
    pool.close()
    pool.join()
