import numpy as np
from skimage import measure
from keras.utils import to_categorical

def getDist(x, y, xys):
    """ calculate the Euclidean distance between a specific point and multi points

        # Args:
            x: float, x of the specific point
            y: float, y of the specific
            xys: 2-dim array, like[[x1, y1], [x2, y2], ... [xn, yn]]

        # Returns: 1-dim array of Euclidean distances
    """
    return np.sqrt(np.square(xys[:,0]-x) + np.square(xys[:, 1]-y))


def generateBoundaryDist(img, R=20, K=10, one_hot=True):
    """ generate the distance to boundary

    # Args:
        img: 2-dim array, input image to detect edges
        R: int, maximum distance
        K: int, bins to classify the distances
        ont-hot: Boolean, whether to transform the result to one-hot encoding.
                 the shapes of the output are (h, w, K) and (h, w), respectively

    # Returns: 2-dim or 3-dim array, the image that represents the distance to boundaries.
    """
    if img.ndim != 2:
        raise InterruptedError("input image must be 2-dim!")

    # a contour is represented as a 2-dim array (y, x)
    # multi contours composed a list of arrays
    contours = measure.find_contours(img, 0.5)

    # get all the points in the boundary
    boundaries = list()
    for contour in contours:
        boundaries.extend(list(contour))
    boundaries = np.array(boundaries)

    # split the range`(-R, R]` into K parts, each one is equal to 2R/K
    per_bin = 2*R/K
    dist_img = np.zeros((img.shape[0], img.shape[1], K), dtype=np.float)
    # if there are no boundaries, set all to 0
    if(len(boundaries)==0):
        return dist_img

    for i in range(dist_img.shape[0]):
        for j in range(dist_img.shape[1]):
            dist = np.sign(img[i,j]-0.5)*np.min(getDist(i, j, boundaries))
            if abs(dist)>=R:
                dist = np.sign(dist)*(R-1e-6)
            dist_img[i,j] = to_categorical(np.floor((dist+R)/per_bin), K)

    if not one_hot:
        dist_img = np.argmax(dist_img, axis=-1)
    return dist_img


def test_unit(fn):
    from keras.preprocessing.image import load_img, img_to_array
    import matplotlib.pyplot as plt
    img = img_to_array(load_img(fn, color_mode="grayscale"))[:,:,0]
    plt.subplot(3,4,1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    R = 20
    K = 10
    dist_img = generateBoundaryDist(img, R, K, False)
    plt.subplot(3, 4, 2)
    plt.imshow(dist_img)
    plt.xticks([])
    plt.yticks([])
    dist_img = generateBoundaryDist(img, R=R, K=K)
    for i in range(K):
        plt.subplot(3, 4, i+3)
        plt.imshow(dist_img[:,:,i])
        plt.title("({})".format(i+1))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == "__main__":
    test_unit("./data_test/0.png")