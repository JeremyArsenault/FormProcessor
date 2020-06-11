import processing
import cv2
import numpy as np
import json

# return relative location on document
def rel_pos(pt, source_dim, dest_dim):
    return int(pt * dest_dim / source_dim)

# return image of field box
def get_field_img(doc, field, WIDTH, HEIGHT):
    xmax = rel_pos(field['xmax'], WIDTH, doc.shape[1])
    xmin = rel_pos(field['xmin'], WIDTH, doc.shape[1])
    ymax = rel_pos(field['ymax'], HEIGHT, doc.shape[0])
    ymin = rel_pos(field['ymin'], HEIGHT, doc.shape[0])

    field = doc[ymin:ymax, xmin:xmax]

    return field

"""
Return dictionary object contatining processed document
doc: grayscale numpy image repr
layout: dict contatining each element and its location on the paper
"""
def process_page(page, page_name, layout):
    try:
        WIDTH = layout['WIDTH']
        HEIGHT = layout['HEIGHT']
    except:
        raise Exception('Invalid layout: WIDTH/HEIGHT')

    processing.init_model()

    # dict contatining processed document
    data = dict()

    for field in layout[page_name].keys():

        print(field)

        field_img = get_field_img(doc, layout[page_name][field], WIDTH, HEIGHT)

        if 'checkbox' in field:
            value, conf = processing.process_checkbox(field_img)
            data[field] = {'value':value,
                           'proba':conf}
        elif 'large-field' in field:
            value, conf = processing.process_large_text_field(field_img)
            data[field] = {'value':value,
                           'proba':conf}         
        else:
            value, conf = processing.process_text_field(field_img)
            data[field] = {'value':value,
                           'proba':conf}

    processing.reset_session()

    return data
    
if __name__=='__main__':
    from pdf2image import convert_from_path

    doc = cv2.cvtColor(np.array(convert_from_path("sample_forms/Adult-Medical-Form-p1-test.pdf")[0]), cv2.COLOR_BGR2GRAY)

    f = open('form_layouts/Adult-Medical-Form.json')
    layout = json.load(f)

    data = process_page(doc, 'page-0', layout)

    for key in data.keys():
        print('Field: ', key)
        print('Transcription: ', data[key]['value'])
        print('Proba: ', data[key]['proba'])

