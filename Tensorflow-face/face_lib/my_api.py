#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-6-13 下午11:13
@email: lph0729@163.com  

"""
import os
import cv2
import random

from face_lib.align_dlib import AlignDlib

size = 96


class GetAlignedFace(object):
    def __init__(self, input_dir, out_dir):
        self.PREDICTOR_PATH = "./face_lib/shape_predictor_68_face_landmarks.dat"
        self.input_dir = input_dir
        self.out_dir = out_dir
        self.pic_names = self.read_pic_names()

    def read_pic_names(self):
        pic_names = []
        for filename in os.listdir(self.input_dir):
            pic_names.append(filename)
        return pic_names

    def read_photo(self, file, num):
        detector = AlignDlib(self.PREDICTOR_PATH)
        path = self.input_dir + "/" + file
        file_name = self.out_dir + "/" + str(num) + "_" + file
        print("--------输出的图片-----------", file_name)

        if not os.path.exists(file_name):
            os.makedirs(file_name)

        index = 1
        for picname in os.listdir(path):
            if picname.endswith(".jpg"):
                image_path = path + "/" + picname
                pic_bgr = cv2.imread(image_path)
                pic_rgb = cv2.cvtColor(pic_bgr, cv2.COLOR_BGR2RGB)
                face_align = detector.align(96, pic_rgb)  # pic_align_rgb dtype:numpy.ndarrray shape:(96, 96, 3)
                if face_align is None:
                    continue
                face_align = cv2.cvtColor(face_align, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_name + "/" + "_" + str(index) + ".jpg", face_align)
                index += 1


class Traversal(object):
    @staticmethod
    def get_triplet_data(path):
        max_num = len(os.listdir(path))
        faces_arrray = [[] for n in range(max_num)]
        id_array = [[] for n in range(max_num)]

        for i, filename in enumerate(os.listdir(path)):
            file_name = path + filename
            for j, image_name in enumerate(os.listdir(file_name)):
                if image_name.endswith(".jpg"):
                    image_path = file_name + "/" + image_name
                    img = cv2.imread(image_path)
                    faces_arrray[i].append(img.astype("float32") / 255.0)
                    id_array[i].append(j)
            # print("-----------第{}次循环:face_array:{},id_array:{}-------------".format(i, faces_arrray, id_array))

        return max_num, faces_arrray, id_array

    @staticmethod
    def generate_train_data(self):
        pass


class Random(object):
    @staticmethod
    def generate_train_data(id_array, num):
        train_data = []
        per_data = []
        p = None

        for i in range(num):
            per_data.clear()
            temp_x = id_array[i]
            if len(temp_x) == 1:
                per_data.append(str(i)+"_"+str(temp_x[0]))
                per_data.append(str(i)+"_"+str(temp_x[0]))
            elif len(temp_x) >= 2:
                random.shuffle(temp_x)
                per_data.append(str(i)+"_"+str(temp_x[0]))
                per_data.append(str(i)+"_"+str(temp_x[1]))

            else:
                continue

            flag = True
            while flag:
                p = random.randint(0, num-1)
                if p != i and len(id_array[p]) != 0:
                    flag = False
            temp_y = id_array[p]
            random.shuffle(temp_y)
            per_data.append(str(p) + "_" + str(temp_x[0]))

            train_data.append(per_data)
        random.shuffle(train_data)
        return train_data
























