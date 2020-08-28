import os
import pickle
from PIL import Image
import json
import argparse
import sys
import cv2

# fix for local importing
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from pose import *
from DFLJPG import DFLJPG
from DFLPNG import DFLPNG


def get_file_list(input_dir):
    """ Return list of images at specified location """
    result = []
    extensions = [".jpg", ".png", ".jpeg"]
    for root, _, files in os.walk(input_dir + '/'):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                result.append(os.path.join(root, file))
        break
        
    return result

def to_JSON(filename, data):
  out = dict()
  out['name'] = filename
  #print (data.get_xseg_mask().tostring())
  if os.path.splitext(filename)[-1].lower() in [".jpg", ".jpeg"] and len(data.get_dict()) == 0:
      data = None
      
  pose = get_pose(filename, data.get_landmarks())

  if data is None:
    out['type'] = ''
    out['parentName'] = ''
    out['landmarks'] = ''
    out['sourceLandmarks'] = ''
    out['matrix'] = ''
    out['x'] = ''
    out['y'] = ''
    out['w'] = ''
    out['h'] = ''
    out['iePolys'] = ''
    out['eyebrowsExpand'] = ''
    out['pose'] = ''
  else:
    out['type'] = data.get_face_type()
    out['parentName'] = data.get_source_filename()
    out['landmarks'] = [[round(float(sublist[0]), 2), round(float(sublist[1]), 2)] for sublist in data.get_landmarks()]
    out['sourceLandmarks'] = [[round(float(sublist[0]), 2), round(float(sublist[1]), 2)] for sublist in data.get_source_landmarks()]
    out['x'] = int(data.get_source_rect()[0])
    out['y'] = int(data.get_source_rect()[1])
    out['w'] = int(data.get_source_rect()[2] - data.get_source_rect()[0])
    out['h'] = int(data.get_source_rect()[3] - data.get_source_rect()[1])
    out['matrix'] = [item for sublist in data.get_image_to_face_mat() for item in sublist]
    out['iePolys'] = get_ie_polys(data)
    out['eyebrowsExpand'] = float(data.get_eyebrows_expand_mod()) if data.get_eyebrows_expand_mod() != None else 1.0
    out['xsegMask'] = get_xseg_mask_data(data)
    out['pose'] = pose
    # print (data.get_xseg_mask())
  out_json = json.dumps(out)

  return out_json

def get_xseg_mask_data(data):
    xseg_data = data.get_xseg_mask()
    if xseg_data == None:
        return None

    return xseg_data.hex()

def get_ie_polys(data):
    polys = data.get_seg_ie_polys()
    if polys is None:
        return ''

    polys = polys.dump()
    polys_ser = serialize_seg_poly_to_array(polys)

    return polys_ser

def get_pose(image_path, landmarks):
    im = Image.open(image_path)
    width, height = im.size
    pose = estimate_pitch_yaw_roll(landmarks, width)

    return pose


def serialize_seg_poly_to_array(data):
    polys_list = []
    for poly in data['polys']:
        poly_points = []
        for point in poly['pts']:
            poly_points.append([round(float(point[0]), 2), round(float(point[1]), 2)])
        polys_list.append([int(poly['type']), poly_points])

    return polys_list

def from_JSON(data):
    json_data = json.loads(data)
    return json_data

def load_folder_data(image_folder):
  files = get_file_list(image_folder)

  for file in files:
      out = load_data(file)
      json_out = to_JSON(file, out)
      # print (json_out)
      sys.stdout.write(json_out + '\n')

def load_data(file):
    if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg"]:
        out = DFLJPG.load(file)
    else:
        out = DFLPNG.load(file)

    return out

def save_data(image_folder):
    files = get_file_list(image_folder) 

    for file in files:
        if os.path.exists(file) == False:
            continue

        data_json  = input(file)
        if (len(data_json) < 2):
            continue
        
        d = from_JSON(data_json)
        file_load = load_data(file)
        ## generate data dic
        dataDic = {}
        dataDic['type'] = d['type'] if d['type'] != '' else None
        dataDic['parentName'] = d['parentName'] if d['parentName'] != '' else None
        dataDic['rect'] = [d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h']] if d['h'] != -1 else None
        dataDic['mat'] = [[d['matrix'][0], d['matrix'][1], d['matrix'][2]], [d['matrix'][3], d['matrix'][4], d['matrix'][5]]] if len(d['matrix']) > 0 else None
        dataDic['landmarks'] = d['landmarks'] if len(d['landmarks']) > 0 else None
        dataDic['sourceLandmarks'] = d['sourceLandmarks'] if len(d['sourceLandmarks']) > 0 else None
        dataDic['eyebrowsExpand'] = d['eyebrowsExpand'] if d['eyebrowsExpand'] > 0 else None
        dataDic['segPolys'] = d['iePolys'] if len(d['iePolys']) > 0 else None

        if (file_load is not None):
                file_load.embed_and_set(file, face_type=dataDic['type'], source_filename=dataDic['parentName'], 
                source_landmarks=dataDic['sourceLandmarks'], source_rect=dataDic['rect'],
                image_to_face_mat=dataDic['mat'], landmarks=dataDic['landmarks'],
                seg_ie_polys=dataDic['segPolys'], eyebrows_expand_mod=dataDic['eyebrowsExpand'])
        else:
            if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg"]:
                DFLJPG.embed_data(file, face_type=dataDic['type'], source_filename=dataDic['parentName'], 
                source_landmarks=dataDic['sourceLandmarks'], source_rect=dataDic['rect'],
                image_to_face_mat=dataDic['mat'], landmarks=dataDic['landmarks'],
                seg_ie_polys=dataDic['segPolys'], eyebrows_expand_mod=dataDic['eyebrowsExpand'])
            else:
                DFLPNG.embed_data(file, face_type=dataDic['type'], source_filename=dataDic['parentName'], 
                source_landmarks=dataDic['sourceLandmarks'], source_rect=dataDic['rect'],
                image_to_face_mat=dataDic['mat'], landmarks=dataDic['landmarks'], eyebrows_expand_mod=dataDic['eyebrowsExpand'])

def update_parent_to_self(image_folder):
    files = get_file_list(image_folder)

    for file in files:
        file_load = load_data(file)
        if file_load is not None:
            file_load.embed_and_set(file, source_filename=file, source_landmarks=file_load.get_landmarks(), source_rect=[0, 0, 256, 256])
            # print (file)
            sys.stdout.write(file + '\n')

def copy_to_file(image_folder, copy_folder):
    files = get_file_list(image_folder)
    for file in files:
        file_load = load_data(file)
        if file_load is not None:
            copy_file = os.path.join(copy_folder, os.path.basename(file))
            pre, ext = os.path.splitext(copy_file)
            copy_file = pre + '.png'
            if (os.path.isfile(copy_file) == False):
                copy_file = pre + '.jpg'
                if (os.path.isfile(copy_file) == False):
                    continue

            if os.path.splitext(copy_file)[1].lower() in [".jpg", ".jpeg"]:
                DFLJPG.embed_data(copy_file, 
                    face_type=file_load.get_face_type(),
                    source_filename=file_load.get_source_filename(),
                    source_landmarks=file_load.get_source_landmarks(),
                    landmarks=file_load.get_landmarks(),
                    source_rect=file_load.get_source_rect(),
                    image_to_face_mat=file_load.get_image_to_face_mat(),
                    eyebrows_expand_mod=file_load.get_eyebrows_expand_mod(),
                    seg_ie_polys=file_load.get_seg_ie_polys(),
                    xseg_mask=file_load.get_xseg_mask()
                    )
            else:
                DFLPNG.embed_data(copy_file, 
                    face_type=file_load.get_face_type(),
                    source_filename=file_load.get_source_filename(),
                    source_landmarks=file_load.get_source_landmarks(),
                    landmarks=file_load.get_landmarks(),
                    source_rect=file_load.get_source_rect(),
                    eyebrows_expand_mod=file_load.get_eyebrows_expand_mod(),
                    image_to_face_mat=file_load.get_image_to_face_mat()
                    )
        sys.stdout.write(file + '\n')

def convert_png_to_jpg(png_path, jpg_path):
    png_images = get_file_list(png_path)

    for file in png_images:
        file_load = load_data(file)
        if file_load is None:
            print ('No file data for file' + file)
            continue

        file_mat = cv2.imread(file)
        pre, ext = os.path.splitext(os.path.basename(file))
        copy_file = os.path.join(jpg_path, pre + '.jpg')
        cv2.imwrite(copy_file, file_mat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        DFLJPG.embed_data(copy_file, 
            face_type=file_load.get_face_type(),
            source_filename=file_load.get_source_filename(),
            source_landmarks=file_load.get_source_landmarks(),
            landmarks=file_load.get_landmarks(),
            source_rect=file_load.get_source_rect(),
            image_to_face_mat=file_load.get_image_to_face_mat()
            )
        print ('Done: ' + copy_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--utoself", action='store_true') # updates file name to self dont use
    parser.add_argument("--copy")
    parser.add_argument("--jpg_folder")
    args = parser.parse_args()

    if args.jpg_folder is not None:
        convert_png_to_jpg(args.input_folder, args.jpg_folder)
        exit(0)

    if args.utoself == True:
        update_parent_to_self(args.input_folder)
        exit(0)

    if args.copy is not None:
        copy_to_file(args.input_folder, args.copy)
        exit(0)

    if args.save == True:
        save_data(args.input_folder)
    else:
        load_folder_data(args.input_folder)