import cv2
import argparse
import sys
import operator
import os

from tqdm import tqdm
from time import sleep


out = sys.stdout

def sort_hist_dissim(path):
    """ Sort by histigram of face dissimilarity """
    input_dir = path

    print("status%Calculating histogram")

    # img_list = [
    #     [img, cv2.calcHist([cv2.imread(img)], [0], None, [256], [0, 256])]
    #     for img in
    #     tqdm(find_images(input_dir), desc="Loading", file=sys.stdout)
    # ]
    img_list = []
    for i, x in enumerate(find_images(input_dir)):
        print (str(i + 1))
        img = cv2.imread(x)
        img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                             cv2.calcHist([img], [1], None, [256], [0, 256]),
                             cv2.calcHist([img], [2], None, [256], [0, 256])
                            , 0])

    print("status%Sorting by histogram dssim")

    img_list_len = len(img_list)
    for i in range(0, img_list_len):
        print (str(i + 1))
        score_total = 0
        for j in range(0, img_list_len):
            if i == j:
                continue
            score_total += cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][3], img_list[j][3], cv2.HISTCMP_BHATTACHARYYA)

        img_list[i][4] = score_total

    img_list = sorted(img_list, key=operator.itemgetter(4), reverse=True)

    return img_list

def sort_hist(path):
    """ Sort by histogram of face similarity """
    input_dir = path

    print("status%Calculating histogram")

    # img_list = [
    #     [img, cv2.calcHist([cv2.imread(img)], [0], None, [256], [0, 256])]
    #     for img in
    #     tqdm(find_images(input_dir), desc="Loading", file=sys.stdout)
    # ]
    img_list = []
    for i, x in enumerate(find_images(input_dir)):
        print (str(i + 1))
        img = cv2.imread(x)
        img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                             cv2.calcHist([img], [1], None, [256], [0, 256]),
                             cv2.calcHist([img], [2], None, [256], [0, 256])
                            ])

    print("status%Sorting by histogram")

    img_list_len = len(img_list)
    for i in range(0, img_list_len - 1):
        print (str(i + 1))
        min_score = float("inf")
        j_min_score = i + 1
        for j in range(i+1,len(img_list)):
            score = cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA) + \
                    cv2.compareHist(img_list[i][3], img_list[j][3], cv2.HISTCMP_BHATTACHARYYA)
            # print (score)
            if score < min_score:
                min_score = score
                j_min_score = j
        img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]

    return img_list

def get_avg_score_hist(img1, references):
    """ Return the average histogram score between a face and
        reference image """
    scores = []
    for img2 in references:
        # score = cv2.compareHist(img1, img2, cv2.HISTCMP_BHATTACHARYYA)
        score = cv2.compareHist(img1[0], img2[0], cv2.HISTCMP_BHATTACHARYYA) + \
                cv2.compareHist(img1[1], img2[1], cv2.HISTCMP_BHATTACHARYYA) + \
                cv2.compareHist(img1[2], img2[2], cv2.HISTCMP_BHATTACHARYYA)
        scores.append(score)
        # print (score)
    return sum(scores) / len(scores)

def group_hist(img_list, min_threshold = 0.3):
    # Groups are of the form: group_num -> reference histogram
    reference_groups = dict()
    # Bins array, where index is the group number and value is
    # an array containing the file paths to the images in that group
    bins = []

    print("status%Grouping histogram")

    img_list_len = len(img_list)
    # reference_groups[0] = [img_list[0][1]]
    reference_groups[0] = [[img_list[0][1], img_list[0][2], img_list[0][3]]]
    bins.append([img_list[0][0]])

    for i in range(1, img_list_len):
        print (str(i + 1))
        current_best = [-1, float("inf")]
        for key, value in reference_groups.items():
            score = get_avg_score_hist([img_list[i][1], img_list[i][2], img_list[i][3]], value)
            if score < current_best[1]:
                current_best[0], current_best[1] = key, score

        if current_best[1] < min_threshold:
            reference_groups[current_best[0]].append([img_list[i][1], img_list[i][2], img_list[i][3]])
            bins[current_best[0]].append(img_list[i][0])
        else:
            reference_groups[len(reference_groups)] = [[img_list[i][1], img_list[i][2], img_list[i][3]]]
            bins.append([img_list[i][0]])

    return bins

def cal_hist(image_file, padding):
    image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    if padding > 0:
        image = image[padding: image.shape[0] - padding, padding: image.shape[1] - padding]
    calc = cv2.calcHist(image, [0], None, [256], [0, 256])

    return calc

def find_images(input_dir):
    """ Return list of images at specified location """
    result = []
    extensions = [".jpg", ".png", ".jpeg"]
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                result.append(os.path.join(root, file))
    return result
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--padding", help="Face padding in pixels")
    parser.add_argument("--sort", action='store_true', help="Sort faces")
    parser.add_argument("--dssim", action='store_true', help="Sort faces")
    parser.add_argument("--group", action='store_true', help="Group faces")

    args = parser.parse_args()

    input_folder = args.input_folder
    if args.padding is None:
        padding = 0
    else:
        padding = int(args.padding)

    if args.sort == False:
        images = find_images(input_folder)
        for image in images:
            hist_score = cal_hist(image, padding)
            _, filename = os.path.split(image)
            print (filename + "%" + str(hist_score.flatten()))      
        sys.exit()

    if args.dssim:
        sort_results = sort_hist_dissim(input_folder)
    else:
        sort_results = sort_hist(input_folder)

    # print group
    if args.group == False:
        print ("status%results")
        result = ''
        for i in range(0, len(sort_results)):
            src = sort_results[i] if isinstance(sort_results[i], str) else sort_results[i][0]
            src_basename = os.path.basename(src)
            result += ',' + src_basename + "%"

        sleep(0.1) # sleep 100ms to ensure it doesn't get combine with print statements
        out.write(result)
        sys.exit()


    group_results = group_hist(sort_results)
    print ("status%results")
    result = ''
    for i in range(0, len(group_results)):
        for j in range(0, len(group_results[i])):
            src = group_results[i][j]
            src_basename = os.path.basename(src)
            result += ',' + src_basename + "%" + str(i)
    sleep(0.1) # sleep 100ms to ensure it doesn't get combine with print statements
    out.write(result)

    

    



