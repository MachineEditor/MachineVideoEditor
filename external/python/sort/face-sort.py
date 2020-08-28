#!/usr/bin python3
# code taken from faceswap github
""" VGG_Face2 inference
Model exported from: https://github.com/WeidiXie/Keras-VGGFace2-ResNet50
which is based on: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/

Licensed under Creative Commons Attribution License.
https://creativecommons.org/licenses/by-nc/4.0/
"""

import sys
import os
import tqdm
import math

import cv2
import numpy as np
from fastcluster import linkage
import keras
import keras.backend as K
from keras.engine import Layer
import argparse
from time import sleep
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from DFLJPG import DFLJPG
from DFLPNG import DFLPNG

class L2_normalize(Layer):
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(L2_normalize, self).__init__(**kwargs)

    def call(self, x):
        return K.l2_normalize(x, self.axis)

    def get_config(self):
        config = super(L2_normalize, self).get_config()
        config["axis"] = self.axis
        return config

class VGGFace2():
    """ VGG Face feature extraction.
        Input images should be in BGR Order """

    def __init__(self, backend="GPU", loglevel="INFO"):
        backend = backend.upper()
        git_model_id = 10
        model_filename = ["vggface2_resnet50_v2.h5"]
        self.input_size = 224
        # Average image provided in https://github.com/ox-vgg/vgg_face2
        self.average_img = np.array([91.4953, 103.8827, 131.0912])

        self.model = self.get_model(git_model_id, model_filename, backend)

    # <<< GET MODEL >>> #
    def get_model(self, git_model_id, model_filename, backend):
        """ Check if model is available, if not, download and unzip it """
        root_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        # print (root_path)
        model_path = os.path.join(root_path, 'models', 'vggface2_resnet50_v2.h5')
        # cache_path = os.path.join(root_path, "plugins", "extract", "recognition", ".cache")
        # model = GetModel(model_filename, cache_path, git_model_id).model_path
        # if backend == "CPU":
        #     if os.environ.get("KERAS_BACKEND", "") == "plaidml.keras.backend":
        #         os.environ["KERAS_BACKEND"] = "tensorflow"
        import keras
        # from lib.model.layers import L2_normalize
        # if backend == "CPU":
        #     with keras.backend.tf.device("/cpu:0"):
        #         return keras.models.load_model(model, {
        #             "L2_normalize":  L2_normalize
        #         })
        # else:
        return keras.models.load_model(model_path, {
            "L2_normalize":  L2_normalize
        })

    def predict(self, face):
        """ Return encodings for given image from vgg_face """
        if face.shape[0] != self.input_size:
            face = self.resize_face(face)
        face = face[None, :, :, :3] - self.average_img
        preds = self.model.predict(face)
        return preds[0, :]

    def resize_face(self, face):
        """ Resize incoming face to model_input_size """
        sizes = (self.input_size, self.input_size)
        interpolation = cv2.INTER_CUBIC if face.shape[0] < self.input_size else cv2.INTER_AREA
        face = cv2.resize(face, dsize=sizes, interpolation=interpolation)
        return face

    @staticmethod
    def find_cosine_similiarity(source_face, test_face):
        """ Find the cosine similarity between a source face and a test face """
        var_a = np.matmul(np.transpose(source_face), test_face)
        var_b = np.sum(np.multiply(source_face, source_face))
        var_c = np.sum(np.multiply(test_face, test_face))
        return 1 - (var_a / (np.sqrt(var_b) * np.sqrt(var_c)))

    def sorted_similarity(self, predictions, method="ward"):
        """ Sort a matrix of predictions by similarity Adapted from:
            https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
        input:
            - predictions is a stacked matrix of vgg_face predictions shape: (x, 4096)
            - method = ["ward","single","average","complete"]
        output:
            - result_order is a list of indices with the order implied by the hierarhical tree

        sorted_similarity transforms a distance matrix into a sorted distance matrix according to
        the order implied by the hierarchical tree (dendrogram)
        """
        num_predictions = predictions.shape[0]
        result_linkage = linkage(predictions, method=method, preserve_input=False)
        np.set_printoptions(suppress=True)
        # print (num_predictions)
        # print (result_linkage)
        # print (result_linkage[0])
        # result_order = self.seriation(result_linkage,
        #                               num_predictions,
        #                               num_predictions + num_predictions - 2)
        result_order = self.seriation_non_recursive(result_linkage, num_predictions)
        # print (result_order)
        return result_order

    def seriation(self, tree, points, current_index):
        """ Seriation method for sorted similarity
            input:
                - tree is a hierarchical tree (dendrogram)
                - points is the number of points given to the clustering process
                - current_index is the position in the tree for the recursive traversal
            output:
                - order implied by the hierarchical tree

            seriation computes the order implied by a hierarchical tree (dendrogram)
        """
        if current_index < points:
            return [current_index]
        node_index = current_index-points
        left = int(tree[node_index, 0])
        right = int(tree[node_index, 1])
        return self.seriation(tree, points, left) + self.seriation(tree, points, right)

    def seriation_non_recursive(self, tree, points, group_threshold = 2.1):
        output = []

        start_ind = points * 2 - 2
        right_nodes_queue = []

        group = []
        group_started = False
        group_right_node_end = -1
        # always go left
        while (len(output) < points):
            # print (output)
            if group_started == True and group_right_node_end == start_ind:
                output.append(group)
                group_started = False
                group_right_node_end = -1
                group = []

            node = start_ind - points
            left = int(tree[node, 0])
            right = int (tree[node, 1])
            distance = float(tree[node, 2])

            if group_started == False and distance <= group_threshold:
                group_started = True
                if len(right_nodes_queue) > 0:
                    group_right_node_end = right_nodes_queue[-1]
                else:
                    group_right_node_end = -1
                # print (str(group_right_node_end) + ' d ' + str(distance))


            # print (str(node) + ' l ' + str(left) + ' r ' + str(right) + ' d ' + str(distance))

            new_ind = -1
            if left < points:
                # output.append(left)

                if group_started == True:
                    group.append(left)
                else:
                    output.append([left])
            else:
                new_ind = left

            if right < points:
                # output.append(right)

                if group_started == True:
                    group.append(right)
                else:
                    output.append([right])
            else:
                if new_ind == -1:
                    new_ind = right
                else:
                    right_nodes_queue.append(right)


            # if len(output) == points:
            #     if group_started == True and len(group) > 0:
            #         output.append(group)

            #     return output

            if new_ind == -1:
                # if right nodes empty it should finished
                if len(right_nodes_queue) == 0:
                    if group_started == True and len(group) > 0:
                        output.append(group)

                    return output

                new_ind = right_nodes_queue.pop()

            start_ind = new_ind
        
        return output

            

def find_images(input_dir):
    """ Return list of images at specified location """
    result = []
    extensions = [".jpg", ".png", ".jpeg"]
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                result.append(os.path.join(root, file))
        break
    return result

# ported original code from https://github.com/YuvalNirkin/find_face_landmarks/blob/master/sequence_face_landmarks/utilities.cpp
# hacked in, written in bad python
def crop_face_by_landmarks(landmarks, img):
    xmin = 100000 # random high numbers
    ymin = 10000
    xmax = -1
    ymax = -1
    sumx = 0
    sumy = 0
    frameWidth = img.shape[1]
    frameHeight = img.shape[0]

    for point in landmarks:
        xmin = min(xmin, point[0])
        ymin = min(ymin, point[1])
        xmax = max(xmax, point[0])
        ymax = max(ymax, point[1])

        sumx += point[0]
        sumy += point[1]

    xmax = round(xmax)
    ymax = round(ymax)
    xmin = round(xmin)
    ymin = round(ymin)

    width = xmax - xmin + 1
    height = ymax - ymin + 1
    centerX = math.floor((xmin + xmax) / 2)
    centerY = math.floor((ymin + ymax) / 2)
    avgX = round(sumx / len(landmarks))
    avgY = round(sumy / len(landmarks))
    devX = centerX - avgX
    devY = centerY - avgY
    dLeft = round(0.1 * width) + abs(devX if devX < 0 else 0)
    dTop = round(height * (max(width / height, 1) * 2 -1)) + abs(devY if devY < 0 else 0)
    dRight = round(0.1 * width) + abs(devX if devX > 0 else 0)
    dBottom = round(0.1 * height) + abs(devY if devY > 0 else 0)

    # Limit to frame boundaries
    # xmin = max(0, xmin - dLeft)
    # ymin = max(0, ymin - dTop)
    # xmax = min(frameWidth -1, xmax + dRight)
    # ymax = min(frameHeight -1, ymax + dBottom)

    # crop is from the chin to the eyebrows
    # expand by 30% of the face to include forehead
    face_height = ymax - ymin
    face_expand = round(face_height * 0.3)
    ymin -= face_expand

    # make square
    sqWidth = int(max(xmax - xmin + 1, ymax - ymin + 1))
    centerX = math.floor((xmin + xmax) / 2)
    centerY = math.floor((ymin + ymax) / 2)
    xmin = centerX - math.floor((sqWidth - 1) / 2)
    ymin = centerY - math.floor((sqWidth - 1) / 2)
    xmax = xmin + sqWidth - 1
    ymax = ymin + sqWidth - 1

    # bounds
    xmin_crop = max(0, xmin)
    ymin_crop = max(0, ymin)
    xmax_crop = min(frameWidth - 1, xmax)
    ymax_crop = min(frameHeight - 1, ymax)

    # print ('x min ' + str(xmin) + ' x max ' + str(xmax) + ' y min ' + str(ymin) + ' y max ' + str(ymax) + ' fw ' + str(frameWidth) + ' fh ' + str(frameHeight))
    # print ('x min crop ' + str(xmin_crop) + ' x max crop ' + str(xmax_crop) + ' y min crop ' + str(ymin_crop) + ' y max crop' + str(ymax_crop))

    # crop
    img_crop = img[ymin_crop: ymax_crop, xmin_crop: xmax_crop]

    # padding
    top = abs(ymin) if ymin < 0 else 0
    left = abs(xmin) if xmin < 0 else 0

    right = xmax - frameWidth + 1 if frameWidth - 1 < xmax else 0
    bottom = ymax - frameHeight + 1 if frameHeight - 1 < ymax else 0

    # print ('top ' + str(top) + ' bottom ' + str(bottom) + ' left ' + str(left) + ' right ' + str(right))
    crop_image_full = cv2.copyMakeBorder(img_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)

    # cv2.imshow('Window', crop_image_full)
    # key = cv2.waitKey(1500)#pauses for 1.5 seconds before fetching next image

    return crop_image_full


def load_embedded_landmarks(image_path):
    embed_data = load_data(image_path)
    if embed_data is None:
        sys.stdout.write('%nodata')
        raise Exception('No image data on file ' + image_path)

    return embed_data.get_landmarks()

def load_data(file):
    if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg"]:
        out = DFLJPG.load(file)
    else:
        out = DFLPNG.load(file)

    return out

def load_image(image_path):
    img = cv2.imread(image_path)
    image_landmarks = load_embedded_landmarks(image_path)

    crop_image = crop_face_by_landmarks(image_landmarks, img)

    # return img
    return crop_image
  

def sort_face(model, input_dir):
    """ Sort by identity similarity """
    image_list = find_images(input_dir)

    sys.stdout.write('%Loading\n')

    preds = np.array([model.predict(load_image(img))
                        for img in tqdm.tqdm(image_list, desc="Calculating...", file=sys.stdout)])

    sys.stdout.write('%Clustering\n')

    indices = model.sorted_similarity(preds, method="ward")
    # img_list = np.array(image_list)[indices]
    face_groups = []
    for group in indices:
        face_group = []
        for indice in group:
            face_group.append(os.path.basename(image_list[indice]))

        face_groups.append(face_group)

    return face_groups

def output_list(sorted_list):
    sys.stdout.write('%Sorting\n')
    sleep(0.1) # sleep 100ms to ensure it doesn't get combine with print statements
    for item in sorted_list:
        os.sys.stdout.write(os.path.basename(item) + '\n')

def output_groups(face_groups):
    sys.stdout.write('%Sorting\n')
    sleep(0.1) # sleep 100ms to ensure it doesn't get combine with print statements
    for group in face_groups:
        sleep(0.2)
        sys.stdout.write('%Group\n')
        sleep(0.2)
        for item in group:
            os.sys.stdout.write(item + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    args = parser.parse_args()

    input_folder = args.input_folder

    model = VGGFace2()
    sorteld_list = sort_face(model, input_folder)
    
    # print (sorteld_list)
    # output_list(sorteld_list)
    output_groups(sorteld_list)