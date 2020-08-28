import os
import pickle
from PIL import Image
import json
import argparse

def get_file_list(input_dir):
    """ Return list of images at specified location """
    result = []
    extensions = [".jpg", ".png", ".jpeg"]
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                result.append(os.path.join(root, file))
    return result

def get_dfl_alignment(filename):
    """ Process the alignment of one face """
    ext = os.path.splitext(filename)[1]

    if ext.lower() in (".jpg", ".jpeg"):
        img = Image.open(filename)
        try:
            dfl_alignments = pickle.loads(img.app["APP15"])
            dfl_alignments["source_rect"] = [n.item()  # comes as non-JSONable np.int32
                                                for n in dfl_alignments["source_rect"]]
            return dfl_alignments
        #except pickle.UnpicklingError:
        except:
            return None

    with open(filename, "rb") as dfl:
        header = dfl.read(8)
        if header != b"\x89PNG\r\n\x1a\n":
            #print("No Valid PNG header: %s", filename)
            return None
        while True:
            chunk_start = dfl.tell()
            chunk_hdr = dfl.read(8)
            if not chunk_hdr:
                break
            
            try:
                chunk_length, chunk_name = struct.unpack("!I4s", chunk_hdr)
                dfl.seek(chunk_start, os.SEEK_SET)
                if chunk_name == b"fcWp":
                    chunk = dfl.read(chunk_length + 12)
                    retval = pickle.loads(chunk[8:-4])
                    #print("Loaded DFL Alignment: (filename: '%s', alignment: %s",
                    #                filename, retval)
                    return retval
                dfl.seek(chunk_length+12, os.SEEK_CUR)
            except:
                return None
        #print("Couldn't find DFL alignments: %s", filename)

def to_JSON(filename, data):
  # print (data)
  out = dict()
  out['name'] = filename
  if data is None:
    out['parentName'] = ''
    out['landmarks'] = ''
    out['sourceLandmarks'] = ''
    out['matrix'] = ''
    out['x'] = ''
    out['y'] = ''
    out['w'] = ''
    out['h'] = ''
  else:
    out['parentName'] = data['source_filename']
    out['landmarks'] = data['landmarks']
    out['sourceLandmarks'] = data['source_landmarks']
    out['x'] = data['source_rect'][0]
    out['y'] = data['source_rect'][1]
    out['w'] = data['source_rect'][2] - data['source_rect'][0]
    out['h'] = data['source_rect'][3] - data['source_rect'][1]
    out['matrix'] = [item for sublist in data['image_to_face_mat'] for item in sublist]

  out_json = json.dumps(out)

  return out_json

def process(image_folder):
  files = get_file_list(image_folder)

  for file in files:
    data = get_dfl_alignment(file)
    if data is None:
        continue
    out = to_JSON(file, data)

    print (out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    args = parser.parse_args()

    process(args.input_folder)
