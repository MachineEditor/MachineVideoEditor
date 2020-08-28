import cv2
from hashlib import sha1
import sys
import argparse
import os

def cv2_read_img(filename, raise_error=False):
    """ Read an image with cv2 and check that an image was actually loaded.
        Logs an error if the image returned is None. or an error has occured.

        Pass raise_error=True if error should be raised """
    # logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    # logger.trace("Requested image: '%s'", filename)
    success = True
    image = None
    try:
        image = cv2.imread(filename)  # pylint:disable=no-member,c-extension-no-member
        if image is None:
            raise ValueError
    except TypeError:
        success = False
        msg = "Error while reading image (TypeError): '{}'".format(filename)
        # logger.error(msg)
        if raise_error:
            raise Exception(msg)
    except ValueError:
        success = False
        msg = ("Error while reading image. This is most likely caused by special characters in "
               "the filename: '{}'".format(filename))
        # logger.error(msg)
        if raise_error:
            raise Exception(msg)
    except Exception as err:  # pylint:disable=broad-except
        success = False
        msg = "Failed to load image '{}'. Original Error: {}".format(filename, str(err))
        # logger.error(msg)
        if raise_error:
            raise Exception(msg)
    # logger.trace("Loaded image: '%s'. Success: %s", filename, success)
    return image

def hash_encode_image(image, extension):
    """ Encode the image, get the hash and return the hash with
        encoded image """
    img = cv2.imencode(extension, image)[1]  # pylint:disable=no-member,c-extension-no-member
    f_hash = sha1(
        cv2.imdecode(  # pylint:disable=no-member,c-extension-no-member
            img,
            cv2.IMREAD_UNCHANGED)).hexdigest()  # pylint:disable=no-member,c-extension-no-member
    return f_hash

def hash_image_file(filename):
    """ Return an image file's sha1 hash """
    # logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    img = cv2_read_img(filename, raise_error=True)
    img_hash = sha1(img).hexdigest()
    # logger.trace("filename: '%s', hash: %s", filename, img_hash)
    return img_hash

def get_file_list(input_dir):
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
    parser.add_argument("path")
    args = parser.parse_args()

    file_list = get_file_list(args.path)

    for file in file_list:
        hash = hash_image_file(file)
        if hash is None:
            continue
        print (file + "%" + hash)