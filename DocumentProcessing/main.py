import argparse
from pdf2image import convert_from_path
import numpy as np
import os
import json

import htr_model.model as htr

def process_text_alpha(img, divs):
    lex_filter = np.zeros(38)
    lex_filter[10:] = lex_filter[10:]+1
    return process_text(img, divs, lex_filter)

def process_text_num(img, divs):
    lex_filter = np.zeros(38)
    lex_filter[:10] = lex_filter[:10]+1
    lex_filter[36:] = lex_filter[36:]+1
    return process_text(img, divs, lex_filter)

def process_text_alphanum(img, divs):
    lex_filter = np.ones(38)
    return process_text(img, divs, lex_filter)

def process_text(img, divs, lex_filter):
    global htr_model
    lex = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ .'

    char_width = img.shape[1]/divs
    pred_chars = []
    pred_probs = []
    for i in range(divs):
        char_img = img[:, int(i*char_width):int((i+1)*char_width)]
        pred = list(htr.predict(char_img, htr_model) * lex_filter)
        pred_probs.append(max(pred))
        pred_chars.append(lex[pred.index(max(pred))])

    pred = ''.join(pred_chars)
    proba = sum(pred_probs) / len(pred_probs)
    return pred, proba

def process_char(img):
    return [0]*65

def process_checkbox(img):
    return True, 0

# return image of field box
def get_field_img(doc, field):
    xmax = field['xmax']
    xmin = field['xmin']
    ymax = field['ymax']
    ymin = field['ymin']

    field = doc[ymin:ymax, xmin:xmax]

    return field

"""
Return dictionary object contatining processed document
doc: grayscale numpy image repr
layout: dict contatining each element and its location on the paper
"""
def process_page(page, layout):
    # dict contatining processed document
    data = {}

    for field in layout.keys():

        field_img = get_field_img(page, layout[field])

        if layout[field]['type'] == 'text-alpha':
            value, conf = process_text_alpha(field_img, layout[field]['divs'])

        elif layout[field]['type'] == 'text-num':
            value, conf = process_text_num(field_img, layout[field]['divs'])

        elif layout[field]['type'] == 'text-alphanum':
            value, conf = process_text_alphanum(field_img, layout[field]['divs'])

        elif layout[field]['type'] == 'checkbox':
            value, conf = process_checkbox(field_img)
        else:
            raise Exception('Field type not recognized: '+layout[field]['type'])

        data[field] = {'value':value,'proba':conf}

    return data

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Open-source pdf process tool')
    parser.add_argument('-l', '--layout', type=str, help='Path to layout file')
    parser.add_argument('-i', '--input_dir', default='input', type=str, help='Path to input directory')
    parser.add_argument('-o', '--output_dir', default='output', type=str, help='Path to output directory')
    parser.add_argument('-m', '--htr_path', default='htr_model/models/emnist-merge', type=str, help='Path to htr model directory')
    parser.add_argument('-c', '--cbox_path', default='checkbox_model/models/cbox_cnn', type=str, help='Path to checkbox model directory')

    args = parser.parse_args()

    LAYOUT_PATH = args.layout
    INPUT_DIR  = args.input_dir
    OUTPUT_DIR = args.output_dir
    HTR_PATH = args.htr_path
    CBOX_PATH = args.cbox_path

    try:
        f = open(LAYOUT_PATH)
        layout = json.load(f)
    except:
        raise Exception('Unable to load layout file')

    global htr_model
    try:
        htr_model = htr.load_model(HTR_PATH)
    except:
        raise Exception('Unable to load htr model')

    IMAGE_PATH_LIST = []
    for f in sorted(os.listdir(INPUT_DIR)):
        f_path = os.path.join(INPUT_DIR, f)
        if os.path.isdir(f_path):
            # skip directories
            continue
        # check if it is an image
        try:
            test_img = convert_from_path(f_path)
            IMAGE_PATH_LIST.append(f_path)
        except:
            pass

    for path in IMAGE_PATH_LIST:
        data = {}
        filename = os.path.split(path)[1]
        file = convert_from_path(path)
        for page_num, page in enumerate(file):
            page_name = 'page-'+str(page_num)
            page_data = process_page(np.array(page), layout[page_name])
            data[page_name] = page_data
        output_path = os.path.join(OUTPUT_DIR, filename[:-4]+'.json')
        print(data)
        with open(output_path, 'w') as f:
            json.dump(data, f)


