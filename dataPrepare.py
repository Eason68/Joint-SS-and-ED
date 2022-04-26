from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np


def dataPrepare():
    """
    将原始数据转化为.npy格式
    :return:
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_PATH = os.path.join(BASE_DIR, "data", "Stanford3dDataset_v1.2_Aligned_Version")
    if not os.path.exists(SRC_PATH):
        print("Please download Stanford3dDataset_v1.2_Aligned_Version.zip "
              "from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under ./data/")
        return
    DST_PATH = os.path.join(BASE_DIR, "data", "s3dis_data")
    if not os.path.exists(DST_PATH):
        os.mkdir(DST_PATH)
    Object_dict = {'clutter': 0, 'ceiling': 1, 'floor': 2, 'wall': 3, 'beam': 4, 'column': 5, 'door': 6,
                   'window': 7, 'table': 8, 'chair': 9, 'sofa': 10, 'bookcase': 11, 'board': 12}

    Areas = [Area for Area in os.listdir(SRC_PATH) if not ".DS_Store" in Area]
    for Area in Areas:

        Rooms = [Room for Room in os.listdir(os.path.join(SRC_PATH, Area)) if not ".DS_Store" in Room]
        for Room in Rooms:

            point_Room = []

            file_name = os.path.join(DST_PATH, Area + "_" + Room + ".npy")
            if os.path.exists(file_name):
                print(file_name + " exists")
                continue

            print("prepare " + Area + " " + Room)
            Annotations = os.path.join(SRC_PATH, Area, Room, "Annotations")
            Objects = [Object for Object in os.listdir(Annotations) if not ".DS_Store" in Object]
            for Object in Objects:

                point_Object = np.loadtxt(os.path.join(Annotations, Object))  # (N,6)
                label_Object = np.tile([Object_dict[Object.split("_", 1)[0]]], (point_Object.shape[0], 1))  # (N,1)
                point_Room.append(np.concatenate([point_Object, label_Object], axis=1))

            point_Room = np.concatenate(point_Room, axis=0)
            np.save(file_name, point_Room)


if __name__ == '__main__':
    dataPrepare()