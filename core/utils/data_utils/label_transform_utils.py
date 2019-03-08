import numpy as np

def color_to_index(color_array, color_mapping, to_sparse=True):
    """ convert colorful label array to label index
        :param color_array: array of shape (height, width) or (height, width, 3)
            RGB or gray label array, where the pixel value is not the label index
        :param colour_mapping: array of shape (n_class,) or (n_class, 3), where n_class is the [total] class number
            Note: the first element is the color of background (label 0).
            e.g., if colour_mapping=[0, 255], pixel equal to 255 are assigned with 1, otherwise 0
            if colour_mapping=[[0, 0, 0], [255,255,255]], pixel equal to [255, 255, 255] are assigned with 1, otherwise 0
        :param to_sparse: bool, optional, default True
            whether to apply a argmax on the last axis to obtain sparse label array

        :return: array of shape (height, width, n_class+1)
        NOTE: if a pixel value pair is not in the colour_mapping, the value of that pixel in the final one-hot array will be [0, 0, ..., 0]
    """
    assert color_mapping is not None
    if len(color_mapping)<2:
        raise ValueError("Invalid length of color map: {}. Expected >= 2!".format(len(color_mapping)))

    onehot_array = [np.zeros((color_array.shape[0], color_array.shape[1]), dtype=np.uint8)]
    for color in color_mapping[1:]:
        _equal = np.equal(color_array, color)
        onehot_array.append(np.all(_equal, axis=-1).astype(np.uint8))
    onehot_array = np.stack(onehot_array, axis=-1).astype(np.uint8)

    # if the color is not in the colour_mapping, assign 0 to represent background
    all_zeros = np.zeros(len(color_mapping), dtype=np.uint8)
    onehot_array[:, :, 0] = np.where(np.all(np.equal(onehot_array, all_zeros), axis=-1), 1, 0)

    if to_sparse:
        onehot_array = np.argmax(onehot_array, axis=-1).astype(np.uint8)
    return onehot_array


def index_to_color(label_array, color_mapping):
    """ encode the 2-dim label array to colorful images
    :param label_array: array of shape (height, width)
        2-dim label array
    :param color_mapping: array of shape (n_class,) or (n_class, 3)
        refer to the one in function 'color_to_label'

    :return: array of shape(height, width, 3) or (height, width), depending on the dimension of color_mapping
    """
    assert color_mapping is not None
    assert label_array.ndim==2

    color_mapping = np.array(color_mapping)
    if color_mapping.ndim==1 or (color_mapping.ndim==2 and color_mapping.shape[1]==3):
        return color_mapping[label_array.astype(np.uint8)]
    else:
        raise ValueError("Invalid color_mapping shape: {}. Expected to be (n,) or (n, 3)".format(color_mapping.shape))