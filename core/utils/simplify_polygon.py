import cv2
import matplotlib.pyplot as plt
import numpy as np

def getAngle(pt1, pt0, pt2):
    """ calculate the angle between 3 points, while the 'pt0' is the vertex
    :param pt1: [x, y]
    :param pt0: [x, y]
    :param pt2: [x, y]
    :return: tne angle, float
    """
    a_vec = np.array([pt1[0]-pt0[0], pt1[1]-pt0[1]])
    b_vec = np.array([pt2[0]-pt0[0], pt2[1]-pt0[1]])
    cos_val = np.matmul(a_vec, b_vec) / (np.linalg.norm(a_vec)*np.linalg.norm(b_vec))
    return np.arccos(cos_val)*180/np.pi

def getLength(pt0, pt1):
    """ calculate the length between two points
    :param pt0: [x, y]
    :param pt1: [x, y]
    :return: length, float
    """
    return np.linalg.norm([pt0[0]-pt1[0], pt0[1]-pt1[1]])



def Polygonization(img, app_epsilon, minArea=10, minLength=3):
    results = list()
    # binary
    ret, img = cv2.threshold(src=img, thresh=0, maxval=1, type=cv2.THRESH_BINARY)
    img = img.astype(np.uint8)

    # find contours
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours[0]:
        approx_plg = cv2.approxPolyDP(cnt, app_epsilon*cv2.arcLength(cnt, True), True)

        # skip polygons with tiny size
        if cv2.contourArea(approx_plg)<minArea:
            continue
        # append the initial two points(e.g, id=0, 1) for calculation convenience
        _approx = approx_plg[:, 0, :].tolist()
        # _approx.extend([[_approx[0][0], _approx[0][1]], [_approx[1][0], _approx[1][1]]])
        #
        # i = 1
        # while i < len(_approx) - 1:
        #     angle = getAngle((_approx[i - 1][0], _approx[i - 1][1]),
        #                      (_approx[i][0], _approx[i][1]),
        #                      (_approx[i + 1][0], _approx[i + 1][1]))
        #     length = getLength((_approx[i - 1][0], _approx[i - 1][1]),
        #                        (_approx[i][0], _approx[i][1]))
        #
        #     if length < minLength or -30 <= angle <= 30:
        #         # print("warning: angle={:.0f}, length={:.2f}".format(angle, length))
        #         del _approx[i]
        #         i -= 1
        #     i += 1

        if len(_approx) > 3:
            results.append(np.array(_approx))
    data = np.zeros_like(img)
    cv2.fillPoly(data, results, 1)
    return data




# def constrainedPolygon(img, simplify_epsilon=0.005, minArea=100):
#     ret, img = cv2.threshold(src=img, thresh=0, maxval=1, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     img = img.astype(np.uint8)
#     contours = cv2.findContours(img, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#
#     plgs = list()
#     for cnt in contours[0]:
#         # approximate polygons
#         epsilon = simplify_epsilon * cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)
#
#         # skip polygons that has small area
#         if cv2.contourArea(cnt)<minArea:
#             continue
#         else:
#             print("shape 1", approx.shape[1])
#             _approx = approx[:, 0, :].tolist()
#             # append the initial two points(e.g, id=0, 1) for calculation convenience
#             _approx.extend([[_approx[0][0], _approx[0][1]], [_approx[1][0], _approx[1][1]]])
#             i = 1
#
#             while i < len(_approx)-1:
#                 angle = getAngle((_approx[i - 1][0], _approx[i - 1][1]),
#                                 (_approx[i][0], _approx[i][1]),
#                                 (_approx[i + 1][0], _approx[i + 1][1]))
#                 length = getLength((_approx[i - 1][0], _approx[i - 1][1]),
#                                 (_approx[i][0], _approx[i][1]))
#
#                 if length < 10 or 0 < angle <= 30:
#                     print("warning: angle={:.0f}, length={:.2f}".format(angle, length))
#                     del _approx[i]
#                     i -= 1
#                 i += 1
#
#             if len(_approx)>3:
#                 plgs.append(np.array(_approx))
#     return plgs


def findCorners(img):
    corners = cv2.goodFeaturesToTrack(img, maxCorners=50, qualityLevel=0.01, minDistance=10)
    return corners




