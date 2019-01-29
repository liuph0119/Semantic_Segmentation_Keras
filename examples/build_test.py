from core.nets import Deeplab_v3p
from core.encoder import resnet_v2_50, resnet_v2_101, resnet_v2_152, resnet_v2_200, resnet_v2_101_separable
from core.encoder.vgg import VGG_19, VGG_16

#model = Deeplab_v3p(input_shape=(256, 256, 3), n_class=1, encoder_name="xception_41", encoder_weights=None)
#encoder = resnet_v2_200((256, 256, 3), True)
encoder = resnet_v2_101_separable((256, 256, 3), kernel_size=5)
encoder.summary()
