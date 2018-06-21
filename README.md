# TensorFlow-FaceRecognition

一.此项目环境是基于 Linux + Python3.5 + Tensorflow开发的一套简单的人脸识别系统

二.项目架构思路分为三步：

    第一：对现有图片的特征检测裁剪并且输出；

    第二：从裁剪输出的图片文件中获取图片的标签数、图片的序列以及图片的内容，然后获取3类别图片集合并且进行对应的模型训练；

    第三：将训练的数据通过matlpotlib展示，一起确认模型的最佳参数；

三.模块介绍：
        1.get_face_align.py: 启动文件， 通过多进程方式对原始图片进行裁剪以及输出；

        2.run_new.py: 一随机的方式产生产生3元组数与类目数，然后进行人脸数据训练。此方式训练时间少，不过损失值不容易收敛，硬件不行的土建此方式训练；

        3.temp: 存放原始的图片数据以及用来测试的测试数据；

        4.train_faces: 存放经过裁剪的人脸图片，用于第二步的人脸模型训练；

        5.face_lib: align_dlib.py、inference.py、my_api.py、shape_predictor_68_face_landmarks.dat

            align_dlib.py： 主要是使用opencv、dlib进行人脸裁剪与对其；

            inference.py：主要是对Siamese神经网络配置

            shape_predictor_68_face_landmarks.dat：人脸检测文件




