import requests
import time
import shutil
import argparse
import json
from bs4 import BeautifulSoup
from pathlib import Path
import os
import sys
from fake_useragent import UserAgent
import urllib.parse as urlparse

import cv2
import numpy as np

image_resolution = '2048x2048'
image_type = '.jpg'
image_counter = 1

getty_watermark_id = 5
wireimage_watermark_id = 125

def get_image_list(url, download_folder, limit):
    global image_counter
    ua = UserAgent(verify_ssl=False)
    headers = {"User-Agent": ua.random}
    source = requests.get(url, headers=headers).content
    soup = BeautifulSoup(str(source).replace('\r\n', ""), "html.parser")
    gallery_list = soup.select('.search-content__gallery-assets')
    if len(gallery_list) == 0 or len(gallery_list) > 1:
        print (soup)
        print (url)
        raise Exception("Gallery is empty")
    gallery = gallery_list[0]
    image_list = soup.select('.gallery-mosaic-asset__container img')
    # print (image_list)
    for image in image_list:
        if image_counter > limit:
            raise Exception("Reached download limit")
        # print (image['src'])
        image_thumbnail_url = image['src']
        image_url = image_thumbnail_url.split('?')[0] + '?s=' + image_resolution
        # print (image_url)
        download_path = os.path.join(download_folder, str(image_counter) + image_type)

        # download single file version
        # try:
        #     save_image(image_url, download_path, headers)
        #     image_counter += 1
        #     print ("Downloaded {}".format(image_url))
        # except Exception as e:
        #     print("[!] Skipping {}".format(image_url))

        getty_download_path = image_url + '&w=' + str(getty_watermark_id)
        try:
            getty_raw = download_image_raw(getty_download_path, headers)
        except Exception as e:
            getty_raw = None

        wireimage_download_path = image_url + '&w=' + str(wireimage_watermark_id)
        try:
            wireimage_raw = download_image_raw(wireimage_download_path, headers)
        except Exception as e:
            wireimage_raw = None

        if wireimage_raw != None and getty_raw != None:
            merged_img = merge_images(getty_raw, wireimage_raw)
            cv2.imwrite(download_path, merged_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        elif wireimage_raw != None:
            with open(download_path, 'wb') as f:
                shutil.copyfileobj(wireimage_raw, f)
        elif getty_raw != None:
            with open(download_path, 'wb') as f:
                shutil.copyfileobj(getty_raw, f)
        else:
            print("[!] Skipping {}".format(image_url))
            continue

        print ("Downloaded {}".format(image_url))
        image_counter += 1


def merge_images(getty_raw, wireimage_raw):
    getty_raw_array = np.asarray(bytearray(getty_raw.read()), dtype="uint8")
    getty_image = cv2.imdecode(getty_raw_array, cv2.IMREAD_COLOR)

    getty_height, getty_width = getty_image.shape[:2]

    wireimage_raw_array = np.asarray(bytearray(wireimage_raw.read()), dtype="uint8")
    wireimage_image = cv2.imdecode(wireimage_raw_array, cv2.IMREAD_COLOR)

    wireimage_height, wireimage_width = wireimage_image.shape[:2]

    getty_height_split = int(getty_height / 5 * 3)
    getty_cropped_img = getty_image[0:getty_height_split]

    wireimage_height_split = int(wireimage_height / 5 * 3)
    wireimage_height_img = wireimage_image[wireimage_height_split:wireimage_height]

    final_image = np.concatenate((getty_cropped_img, wireimage_height_img), axis=0)

    return final_image

def download_image_raw(link, headers):
    r = requests.get(link, stream=True, headers=headers)
    if r.status_code == 200:
        r.raw.decode_content = True
        return r.raw
    else:
        print (r.status_code)
        raise Exception("Image returned a {} error.".format(r.status_code))


def save_image(link, file_path, headers):
    r = requests.get(link, stream=True, headers=headers)
    if r.status_code == 200:
        with open(file_path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        raise Exception("Image returned a {} error.".format(r.status_code))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Give the query that I should parse.")
    parser.add_argument("save_folder", help="Save folder to download images.")
    parser.add_argument("--limit", default=6000) # 60 * 100
    parser.add_argument('--sort', default="best", choices=["best", "oldest", "newest", "mostpopular"])
    parser.add_argument('--people', default=None, choices=["one", "two", "group"], nargs="+")
    parser.add_argument('--composition', default=None, choices=["headshot", "waistup", "fulllength", "threequarterlength", "lookingatcamera", "candid"], nargs="+")
    args = parser.parse_args()

    query = args.query
    save_folder = args.save_folder
    limit = int(args.limit)
    sort = args.sort
    num_people = args.people
    people_composition = args.composition

    query_parsed = query.lower().replace(' ', '-')

    url = 'https://www.gettyimages.com/photos/{}?family=editorial&sort={}'.format(
        str(query_parsed), sort)

    if num_people != None:
        people_query = ','.join(num_people)
        url += '&numberofpeople=' + people_query

    if people_composition != None:
        composition_query = ','.join(people_composition)
        url +=  '&compositions=' + composition_query

    # print (url)
    page = 1
    while page < 100:
        url_page = url + '&page=' + str(page)
        get_image_list(url_page, save_folder, limit)
        page += 1