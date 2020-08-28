import requests
import time
import shutil
import argparse
import json
from bs4 import BeautifulSoup
from pathlib import Path
import lxml.html
import os
import sys
from fake_useragent import UserAgent
if sys.version_info[0] > 2:
    import urllib.parse as urlparse
else:
    import urlparse
    import io
    reload(sys)
    sys.setdefaultencoding('utf8')

'''
Commandline based Bing Images scraping/downloading. Gets unlimited amounts of images.
Author: Rushil Srivastava (rushu0922@gmail.com)
'''


def error(link):
    print("[!] Skipping {}. Can't download or no metadata.\n".format(link))
    # file = Path("dataset/logs/bing/errors.log".format(query))
    # if file.is_file():
    #     with open("dataset/logs/bing/errors.log".format(query), "a") as myfile:
    #         myfile.write(link + "\n")
    # else:
    #     with open("dataset/logs/bing/errors.log".format(query), "w+") as myfile:
    #         myfile.write(link + "\n")


def save_image(link, file_path, headers):
    r = requests.get(link, stream=True, headers=headers)
    if r.status_code == 200:
        with open(file_path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        raise Exception("Image returned a {} error.".format(r.status_code))


def download_image(link, image_data, metadata, path):
    download_image.delta += 1
    # Use a random user agent header for bot id
    ua = UserAgent(verify_ssl=False)
    headers = {"User-Agent": ua.random}

    # Get the image link
    try:
        # Get the file name and type
        file_name = link.split("/")[-1]
        type = file_name.split(".")[-1]
        type = (type[:3]) if len(type) > 3 else type
        if type.lower() == "jpe":
            type = "jpeg"
        if type.lower() not in ["jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
            type = "jpg"

        # Download the image
        print("[%] Downloading Image #{} from {}".format(
            download_image.delta, link))
        try:
            save_image(link, path + "/".format(query) +
                       "Scrapper_bing_{}.{}".format(str(download_image.delta), type), headers)
            print("[%] Downloaded File")
            # if metadata:
            #     with open("dataset/bing/{}/Scrapper_{}.json".format(query, str(download_image.delta)), "w") as outfile:
            #         json.dump(image_data, outfile, indent=4)
        except Exception as e:
            download_image.delta -= 1
            print("[!] Issue Downloading: {}\n[!] Error: {}".format(link, e))
            error(link)
    except Exception as e:
        download_image.delta -= 1
        print("[!] Issue getting: {}\n[!] Error:: {}".format(link, e))
        error(link)


if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Give the query that I should parse.")
    parser.add_argument("save_folder", help="Save folder to download images.")
    parser.add_argument("--limit", help="Total amount of images I should download. Default 1000",
                        type=int, default=1000, required=False)
    parser.add_argument("--json", help="Download image data.",
                        action='store_true', required=False)
    parser.add_argument("--adult-filter-off", help="Disable adult filter",
                        action='store_true', required=False)
    parser.add_argument("--headless", help="Create a headless ChromeDriver instance.",
                        action='store_true', required=False) # not needed, doesn't use chrome
    args = parser.parse_args()

    # set local vars from user input
    query = args.query
    delta = args.limit
    save_folder = args.save_folder
    metadata = args.json if args.json is not None else False
    adult = "off" if args.adult_filter_off else "on"
    url = "https://www.bing.com/images/async?q={}&first=0&adlt={}".format(
        str(query), adult)

    # if not os.path.exists(args.save_folder):
    #     print("Save directory doesn't exists")
    #     exit()
    # check directory and create if necessary
    # os.chdir(os.getcwd())
    # if not os.path.isdir("dataset/"):
    #     os.makedirs("dataset/")
    # if not os.path.isdir("dataset/bing/{}".format(query)):
    #     os.makedirs("dataset/bing/{}".format(query))
    # if not os.path.isdir("dataset/logs/bing/".format(query)):
    #     os.makedirs("dataset/logs/bing/".format(query))

    # set stack limit
    sys.setrecursionlimit(1000000)

    page_counter = 0
    link_counter = 0
    download_image.delta = 0
    while download_image.delta < delta:
        # Parse the page source and download pics
        ua = UserAgent(verify_ssl=False)
        headers = {"User-Agent": ua.random}
        payload = (("q", str(query)), ("first", page_counter), ("adlt", adult))
        source = requests.get(
            "https://www.bing.com/images/async", params=payload, headers=headers).content
        soup = BeautifulSoup(str(source).replace('\r\n', ""), "html.parser")

        # try:
        #     os.remove("dataset/logs/bing/errors.log")
        # except OSError:
        #     pass

        # Get the links and image data
        #print(soup)
        links = [json.loads(i.get("m").replace('\r\n', ""))["murl"]
                 for i in soup.find_all("a", class_="iusc")]
        print("[%] Indexed {} Images on Page {}.".format(
            len(links), page_counter + 1))
        print("\n===============================================\n")
        print("[%] Getting Image Information.")
        images = {}
        for a in soup.find_all("a", class_="iusc"):
            if download_image.delta >= delta:
                break
            print("\n------------------------------------------")
            iusc = json.loads(a.get("m"))
            link = iusc["murl"]
            print("\n[%] Getting info on: {}".format(link))
            try:
                image_data = "bing", query, link, iusc["purl"], iusc["md5"]
                images[link] = image_data
                try:
                    download_image(link, images[link], metadata, save_folder)
                except Exception as e:
                    error(link)
            except Exception as e:
                images[link] = image_data
                print("[!] Issue getting data: {}\n[!] Error: {}".format(link, e))

            link_counter += 1

        page_counter += 1

    print("\n\n[%] Done. Downloaded {} images.".format(download_image.delta))
    print("\n===============================================\n")
