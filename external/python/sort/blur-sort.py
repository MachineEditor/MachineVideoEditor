import argparse
import numpy as np
import cv2
import os
import sys
import math
from tqdm import tqdm
from concurrent import futures

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from DFLJPG import DFLJPG
from DFLPNG import DFLPNG

# from estimate_sharpnes import estimate_sharpness

def find_images(input_dir):
    result = []
    extensions = [".jpg", ".png", ".jpeg"]
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                result.append(os.path.join(root, file))
    
    return result

def estimate_blur(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # try to get hull mask
    # too slow, results are not that good needs debug
    data = load_data(image_file)
    if data is not None:
        image = crop_face_by_landmarks(data.get_landmarks(), image)
    #     face_mask = get_image_hull_mask(image.shape, data.get_landmarks())
    #     image = (image*face_mask).astype(np.uint8)

    blur_map = cv2.Laplacian(image, cv2.CV_32F)
    score = np.var(blur_map)  / np.sqrt(image.shape[0] * image.shape[1])
    # score = np.var(blur_map)
    # score = estimate_sharpness(image)
    
    return score

def load_data(file):
    if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg"]:
        out = DFLJPG.load(file)
    else:
        out = DFLPNG.load(file)

    return out

# landmark crop from face-sort
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

    # expand by 30% of the face to include forehead, not expanding here
    expand = 0
    face_height = ymax - ymin
    face_expand = round(face_height * expand)
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

    # crop
    img_crop = img[ymin_crop: ymax_crop, xmin_crop: xmax_crop]

    # padding
    top = abs(ymin) if ymin < 0 else 0
    left = abs(xmin) if xmin < 0 else 0

    right = xmax - frameWidth + 1 if frameWidth - 1 < xmax else 0
    bottom = ymax - frameHeight + 1 if frameHeight - 1 < ymax else 0

    crop_image_full = cv2.copyMakeBorder(img_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)

    # cv2.imshow('Window', crop_image_full)
    # key = cv2.waitKey(1500)#pauses for 1.5 seconds before fetching next image

    return crop_image_full

# Taken from
# https://github.com/iperov/DeepFaceLab/blob/f4a661b742754d03d87e19280176ebdd6979d7c7/facelib/LandmarksProcessor.py#L363
def get_image_hull_mask(image_shape, image_landmarks):
    hull_mask = np.zeros(image_shape[0:2]+(1,),dtype=np.float32)

    lmrks = expand_eyebrows(image_landmarks)

    r_jaw = (lmrks[0:9], lmrks[17:18])
    l_jaw = (lmrks[8:17], lmrks[26:27])
    r_cheek = (lmrks[17:20], lmrks[8:9])
    l_cheek = (lmrks[24:27], lmrks[8:9])
    nose_ridge = (lmrks[19:25], lmrks[8:9],)
    r_eye = (lmrks[17:22], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    l_eye = (lmrks[22:27], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    nose = (lmrks[27:31], lmrks[31:36])
    parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

    for item in parts:
        merged = np.concatenate(item)
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(merged), (1,) )

    return hull_mask

def expand_eyebrows(lmrks, eyebrows_expand_mod=1.0):
    if len(lmrks) != 68:
        raise Exception('works only with 68 landmarks')
    lmrks = np.array( lmrks.copy(), dtype=np.int )

    # #nose
    ml_pnt = (lmrks[36] + lmrks[0]) // 2
    mr_pnt = (lmrks[16] + lmrks[45]) // 2

    # mid points between the mid points and eye
    ql_pnt = (lmrks[36] + ml_pnt) // 2
    qr_pnt = (lmrks[45] + mr_pnt) // 2

    # Top of the eye arrays
    bot_l = np.array((ql_pnt, lmrks[36], lmrks[37], lmrks[38], lmrks[39]))
    bot_r = np.array((lmrks[42], lmrks[43], lmrks[44], lmrks[45], qr_pnt))

    # Eyebrow arrays
    top_l = lmrks[17:22]
    top_r = lmrks[22:27]

    # Adjust eyebrow arrays
    lmrks[17:22] = top_l + eyebrows_expand_mod * 0.5 * (top_l - bot_l)
    lmrks[22:27] = top_r + eyebrows_expand_mod * 0.5 * (top_r - bot_r)
    return lmrks

# https://github.com/deepfakes/faceswap/blob/619bd415aa2f8aa11d14fd1b74f92a37fa6ad96a/tools/sort/sort.py#L118
def estimate_blur_threading(image_list):
    """ Multi-threaded, parallel and sequentially ordered image loader """
    # logger.info("Loading images...")
    with futures.ThreadPoolExecutor() as executor:
        blur_calc = list(tqdm(executor.map(estimate_blur, image_list),
                                desc="Loading Images...",
                                file=sys.stdout,
                                total=len(image_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")

    args = parser.parse_args()

    input_folder = args.input_folder

    images = find_images(input_folder)
    # estimate_blur_threading(images) # need to setout proper output
    for image in images:
        blur_score = estimate_blur(image)
        _, filename = os.path.split(image)
        print (filename + "%" + str(np.around([blur_score], 4)[0]))