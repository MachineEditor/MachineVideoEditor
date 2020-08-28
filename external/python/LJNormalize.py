# Taken from https://github.com/cubrink/LJNormalize/blob/master/LJNormalize.py
# Filename      :       LJNormalize.py
# Author        :       Curtis Brinker
# Description   :       Normalize and format data in .csv's in a way that
#                       mimics the LJ Speech Dataset. This script is intended
#                       to be used in conjunction with srt-parse.py
#                       (see github.com/cubrink/srt-parse)
#
#                       The LJ Speech Dataset can be accessed at
#                       https://keithito.com/LJ-Speech-Dataset/

import argparse
import inflect
import csv
import re
import os

# Regex pattern to find numbers in strings.
# Accounts for possible commas and decimals
# Not perfect but 'hopefully' transciptions
# shouldn't have that bad of formatting errors
num_pattern = r'[\d]+(?:,?[\d]{3})*(?:\.[\d]+)?'


# 
titles = {r'Mr\.': 'Mister',
            r'Mrs\.': 'Misess',
            r'Dr\.': 'Doctor',
            r'No\.': 'Number',
            r'St\.': 'Saint',
            r'Co\.': 'Company',
            r'Jr\.': 'Junior',
            r'Maj\.': 'Major',
            r'Gen\.': 'General',
            r'Drs\.': 'Doctors',
            r'Rev\.': 'Reverend',
            r'Lt\.': 'Lieutenant',
            r'Hon\.': 'Honorable',
            r'Sgt\.': 'Sergeant',
            r'Capt\.': 'Captain',
            r'Esq\.': 'Esquire',
            r'Ltd\.': 'Limited',
            r'Col\.': 'Colonel',
            r'Ft\.': 'Fort'}

# Do these manually? Also roman numerals are tricky too
# There are obviously more currency symbols. Add more as needed.
troublesome = ['$', '£', '€', '§']

def build_parser():
    '''
    Creates an argparse parser
    '''
    parser = argparse.ArgumentParser(description='Normalize text data in csv similarly to \
                                                  the LJSpeech dataset',
                                     prog='LJNormalize')
    parser.add_argument('input', type=str,
                        help='Location of .csv file to be normalized')
    parser.add_argument('--overwrite', type=bool,
                        help='Write directly to the input file',
                        default=False)
    parser.add_argument('--out-filename', type=str,
                        help='Name of file to write to',
                        default='metadata.csv')
    parser.add_argument('--in-encoding', type=str,
                        help='Encoding to use when reading the .csv')
    parser.add_argument('--out-encoding', type=str,
                        help='Encoding to use when writing to .csv')

    # Note that because the data expected is transcripts commas will be used
    # commonly. Instead we use '|' to seperate the columns
    parser.add_argument('--csv-seperator', type=str,
                        help='Character to used to seperate csv columns',
                        default='|')
    parser.add_argument('--log-troublesome', type=bool,
                        help='Create a log file listing lines with difficult \
                              characters to normalize.',
                        default=True)
    parser.add_argument('--log-name', type=str,
                        help='Name of log file',
                        default='troublesome.log')
    return parser


def read_csv():
    '''
    Returns a list of each row in csv
    Each row represeneted as a list of elements between csv seperator
    '''
    with open(args.input, 'r', encoding=args.in_encoding) as csvfile:
        lines = [line for line in csv.reader(csvfile, delimiter=args.csv_seperator)]
    return lines

def write_csv(lines):
    '''
    Write data to csv formatted similarly to the LJSpeech dataset.
    '''
    # Determine filename
    if args.overwrite:
        filename = args.input
    else:
        filename = args.out_filename

    with open(filename, 'w', encoding=args.out_encoding) as f:
        # Each line formatted as [audio_filename, unnormalized text, normalized text]
        for line in lines:
            f.write(args.csv_seperator.join(line) + '\n')

def normalize_numbers(text, engine):
    '''
    Normalize a string of text by spelling out numbers.
    '''
    slices = [(m.start(), m.end()) for m in re.finditer(num_pattern, text)]
    normalizations = [engine.number_to_words(text[slice(*s)]) for s in slices]
    normalized = text
    for s, norm in zip(slices, normalizations):
        normalized = normalized.replace(text[slice(*s)], norm, 1)
    return normalized
               
    
def normalize_titles(text):
    '''
    Normalize a string of text by spelling out common titles/abbreviations
    '''
    for pattern, replacement in titles.items():
        text = re.sub(pattern, replacement, text)
    return text

def troublesome_log(text, idx):
    '''
    Log to file if a troublesome character is found.
    This lines likely will require manual trancription.
    '''
    for c in troublesome:
        if c in text:
            with open(args.log_name, 'a') as f:
                f.write("Troublesome character {c} on line {idx}", '\n')


def parse_lines(engine):
    print ('ready')
    while True:
        text = input()
        if text == '%end%':
            return

        normalized_text = normalize_numbers(text, engine)
        normalized_text = normalize_titles(normalized_text)
        print(normalized_text)

def parse_line(engine, text):
    normalized_text = normalize_numbers(text, engine)
    normalized_text = normalize_titles(normalized_text)
    print(normalized_text)

if __name__ == "__main__":
# Create parser and parse arguments
# '$', '£', '€', '§' Issues with normalization
    # parser = build_parser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--single',)
    args = parser.parse_args()

    engine = inflect.engine()

    if args.single:
        parse_line(engine, args.single)
    else:
        parse_lines(engine)

    # Read lines from csv
    # lines = read_csv()

    # # Normalize each line. Enter normalized text into second column
    # for idx, line in enumerate(lines):
    #     filename, text = line

    #     normalized_text = normalize_numbers(text)
    #     normalized_text = normalize_titles(normalized_text)
    
    #     if args.log_troublesome:
    #         troublesome_log(text, idx)
    #     lines[idx].append(normalized_text)

    # # Write to file
    # write_csv(lines)
    
