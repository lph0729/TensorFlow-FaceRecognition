#!/usr/bin/env python  
# encoding: utf-8 

""" 
@author: payneLi  
@time: 18-6-14 下午10:31
@email: lph0729@163.com  

"""

import numpy as np
import tensorflow as tf
from time import ctime
import csv

from face_lib import my_api, inference

"""以随机方式训练人脸数据"""

my_faces_path = "./train_faces/"  # 经过特征检测与对齐的人脸数据集目录
model_file = "./model/random/train_faces.model"  # 模型存放路径
size = my_api.size


def get_row_train_data(sess, train_data, saver, train_step, image_array, num, step):
    loss_step = 0
    losses = 0
    image_x_1 = []
    image_x_2 = []
    image_x_3 = []

    for step_i, data in enumerate(train_data):
        if step_i > 1 and loss_step % batch_size == 0:
            loss_step += 1
            train_anc = np.array(image_x_1)
            train_pos = np.array(image_x_2)
            train_neg = np.array(image_x_3)

            loss_v = sess.run(siamese.loss, feed_dict={
                siamese.x_1: train_anc,
                siamese.x_2: train_pos,
                siamese.x_3: train_neg,
                siamese.keep_f: 1.0
            })

            losses += loss_v
            print("time: %s step: %d , %d loss: %.4f losses: %.4f" % (ctime(), step, loss_step, loss_v, losses))

            if loss_step % 100 == 0 and losses <= 0.002:
                saver.save(sess, model_file)
                print("保存成功")

            image_x_1.clear()
            image_x_2.clear()
            image_x_3.clear()

        x_1 = data[0]
        x_2 = data[1]
        x_3 = data[2]
        id_x_1 = x_1.split("_")[0]
        id_y_1 = x_1.split("_")[1]
        id_x_2 = x_2.split("_")[0]
        id_y_2 = x_2.split("_")[1]
        id_x_3 = x_3.split("_")[0]
        id_y_3 = x_3.split("_")[1]

        for i in range(num):
            if int(id_x_1) == i:
                for j, img in enumerate(image_array[i]):
                    if int(id_y_1) == j:
                        image_x_1.append(img)
            if int(id_x_2) == i:
                for j, img in enumerate(image_array[i]):
                    if int(id_y_2) == j:
                        image_x_2.append(img)

            if int(id_x_3) == i:
                for j, img in enumerate(image_array[i]):
                    if int(id_y_3) == j:
                        image_x_3.append(img)
    return losses


def cnn_train():
    l_rate = 0.00005
    train_step = tf.train.AdamOptimizer(l_rate).minimize(siamese.loss)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    new = True
    loss_sum = 0
    with tf.Session() as sess:
        sess.run(init)
        input_var = input("发现模型，是否开始训练[yes/no]?")
        if input_var == "yes":
            new = False

        if not new:
            saver.restore(sess, model_file)

        for step in range(5000):
            # 每次读取50张图片,生成训练数据
            train_data = my_api.Random.generate_train_data(id_array, max_num)  # ['0_2', '0_0', '62_2']
            losses = get_row_train_data(sess, train_data, saver, train_step, np_array, max_num, step)
            print("step: %s  losses: %s  rate: %s" % (step, losses, l_rate))
            loss_sum += losses

            if step > 1 and step % 3 == 0:
                with open("./out/random/print_result.csv", "a+", newline="") as writer_csv:
                    csv_print_writer = csv.writer(writer_csv, dialect="excel")
                    csv_print_writer.writerow([ctime(), "step", step, "loss_sum", loss_sum, "rate", l_rate])
                l_rate = l_rate * 0.8
                loss_sum = 0
                saver.save(sess, model_file)
                print("保存成功")
        saver.save(sess, model_file)
        print("保存成功")


if __name__ == '__main__':
    traversal = my_api.Traversal()
    max_num, faces_array, id_array = traversal.get_triplet_data(my_faces_path)
    # print("标签数为：{} 图片数组为：{} 图片id:{}".format(max_num, faces_array, id_array))
    # print(id_array[0])
    np_array = np.array(faces_array)

    # 将图片分批次训练,每批次50张
    batch_size = 50
    siamese = inference.Siamese(size)
    cnn_train()
