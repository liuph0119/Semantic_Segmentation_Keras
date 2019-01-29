import datetime
import os
import time

from core.utils.predicting_utils import predict_buildingfootprint
from core.utils.model_utils import load_custom_model


if __name__ == "__main__":
    # # # parameters to be modified
    model_path = "E:/SegData/inria/models/mynet_v4_inria.h5"
    model_name = "mynet_v4"
    image_size = 256
    stride = 128
    root_dir = "E:/SegData/inria/data_ori/my_test"
    sub_dirs = ["austin_test", "chicago_test", "kitsap_test", "tyrol-w_test", "vienna_test"]
    # ======================================================================

    print("%s: loading network: %s..." % (datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"), model_path))
    model = load_custom_model(model_path)

    for sub_dir in sub_dirs:
        _dir = os.path.join(root_dir, sub_dir)

        image_names = os.listdir(_dir + "/image")
        t1 = time.time()
        for image_name in image_names:
            image_path = _dir + "/image/" + image_name
            label_path = _dir + "/label/" + image_name
            pred_path = _dir + "/prediction/" + image_name.replace(".tif", "_{}.tif".format(model_name))

            print("%s: predicting image: %s" % (datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"), image_path))
            predict_buildingfootprint(model, image_path, pred_path, img_size=image_size, stride=stride, softmax=0)
        t2 = time.time()
        print("average_time: {}\n\n\n".format((t2-t1)/len(image_names)))