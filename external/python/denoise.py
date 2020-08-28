import os
import sys
import cv2
import argparse

def get_file_list(input_dir):
    """ Return list of images at specified location """
    result = []
    extensions = [".jpg", ".png", ".jpeg"]
    for root, _, files in os.walk(input_dir + '/'):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                result.append(os.path.join(root, file))
    return result

def denoise_image(path, out_folder, h, t_size, s_size):
    img = cv2.imread(path)

    dst = cv2.fastNlMeansDenoisingColored(img, None, h, h, t_size, s_size)

    new_file_path = os.path.join(out_folder, os.path.basename(path))
    ext = os.path.splitext(path)[1].lower()
    if ext == '.jpg':
        cv2.imwrite(new_file_path, dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    elif ext == '.png':
        cv2.imwrite(new_file_path, dst)


def process_denoise(folder, out_folder, h, t_size, s_size, start_idx, end_idx):
    files = get_file_list(folder)

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(files)

    for i in range(start_idx, end_idx):
        denoise_image(files[i], out_folder, h, t_size, s_size)

        sys.stdout.write(os.path.basename(files[i]) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--h")
    parser.add_argument("--t_size") # needs to be odd
    parser.add_argument("--s_size") # needs to be odd
    parser.add_argument("--copy_al", action='store_true') # not implemented
    # parser.add_argument("--start_idx")
    # parser.add_argument("--end_idx")
    args = parser.parse_args()

    process_denoise(args.input_folder, args.output_folder,
        # int(args.h), int(args.t_size), int(args.s_size), int(args.start_idx), int(args.end_idx))
        int(args.h), int(args.t_size), int(args.s_size), None, None)
        