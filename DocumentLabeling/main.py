#!/bin/python
import argparse
import glob
import json
import os
import re

import cv2
import numpy as np
from tqdm import tqdm

from pdf2image import convert_from_path

DELAY = 20 # keyboard delay (in milliseconds)
WITH_QT = False
try:
    cv2.namedWindow('Test')
    cv2.displayOverlay('Test', 'Test QT', 500)
    WITH_QT = True
except cv2.error:
    print('-> Please ignore this error message\n')
cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Open-source image labeling tool')
parser.add_argument('-i', '--input_dir', default='input', type=str, help='Path to input directory')
parser.add_argument('-o', '--output_dir', default='output', type=str, help='Path to output directory')
parser.add_argument('-t', '--thickness', default='3', type=int, help='Bounding box and cross line thickness')

args = parser.parse_args()

class_index = 0
img_index = 0
img = None
img_objects = []

color = (0,0,255)

INPUT_DIR  = args.input_dir
OUTPUT_DIR = args.output_dir

WINDOW_NAME = 'DocumentLabeling'

# selected bounding box
prev_was_double_click = False
is_bbox_selected = False
selected_bbox = -1
LINE_THICKNESS = args.thickness

pdf_page = 0

mouse_x = 0
mouse_y = 0
point_1 = (-1, -1)
point_2 = (-1, -1)

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''


# Check if a point belongs to a rectangle
def pointInRect(pX, pY, rX_left, rY_top, rX_right, rY_bottom):
    return rX_left <= pX <= rX_right and rY_top <= pY <= rY_bottom



# Class to deal with bbox resizing
class dragBBox:
    '''
        LT -- MT -- RT
        |            |
        LM          RM
        |            |
        LB -- MB -- RB
    '''

    # Size of resizing anchors (depends on LINE_THICKNESS)
    sRA = LINE_THICKNESS * 2

    # Object being dragged
    selected_object = None

    # Flag indicating which resizing-anchor is dragged
    anchor_being_dragged = None

    '''
    \brief This method is used to check if a current mouse position is inside one of the resizing anchors of a bbox
    '''
    @staticmethod
    def check_point_inside_resizing_anchors(eX, eY, obj):
        _class_name, x_left, y_top, x_right, y_bottom = obj
        # first check if inside the bbox region (to avoid making 8 comparisons per object)
        if pointInRect(eX, eY,
                        x_left - dragBBox.sRA,
                        y_top - dragBBox.sRA,
                        x_right + dragBBox.sRA,
                        y_bottom + dragBBox.sRA):

            anchor_dict = get_anchors_rectangles(x_left, y_top, x_right, y_bottom)
            for anchor_key in anchor_dict:
                rX_left, rY_top, rX_right, rY_bottom = anchor_dict[anchor_key]
                if pointInRect(eX, eY, rX_left, rY_top, rX_right, rY_bottom):
                    dragBBox.anchor_being_dragged = anchor_key
                    break

    '''
    \brief This method is used to select an object if one presses a resizing anchor
    '''
    @staticmethod
    def handler_left_mouse_down(eX, eY, obj):
        dragBBox.check_point_inside_resizing_anchors(eX, eY, obj)
        if dragBBox.anchor_being_dragged is not None:
            dragBBox.selected_object = obj

    @staticmethod
    def handler_mouse_move(eX, eY):
        if dragBBox.selected_object is not None:
            class_name, x_left, y_top, x_right, y_bottom = dragBBox.selected_object

            # Do not allow the bbox to flip upside down (given a margin)
            margin = 3 * dragBBox.sRA
            change_was_made = False

            if dragBBox.anchor_being_dragged[0] == "L":
                # left anchors (LT, LM, LB)
                if eX < x_right - margin:
                    x_left = eX
                    change_was_made = True
            elif dragBBox.anchor_being_dragged[0] == "R":
                # right anchors (RT, RM, RB)
                if eX > x_left + margin:
                    x_right = eX
                    change_was_made = True

            if dragBBox.anchor_being_dragged[1] == "T":
                # top anchors (LT, RT, MT)
                if eY < y_bottom - margin:
                    y_top = eY
                    change_was_made = True
            elif dragBBox.anchor_being_dragged[1] == "B":
                # bottom anchors (LB, RB, MB)
                if eY > y_top + margin:
                    y_bottom = eY
                    change_was_made = True

            if change_was_made:
                action = "resize_bbox:{}:{}:{}:{}".format(x_left, y_top, x_right, y_bottom)
                edit_bbox(dragBBox.selected_object, action)
                # update the selected bbox
                dragBBox.selected_object = [class_name, x_left, y_top, x_right, y_bottom]

    '''
    \brief This method will reset this class
     '''
    @staticmethod
    def handler_left_mouse_up(eX, eY):
        if dragBBox.selected_object is not None:
            dragBBox.selected_object = None
            dragBBox.anchor_being_dragged = None

def page_name(x):
    return 'page-'+str(x)

def empty_json(img_path):
    shape = np.array(convert_from_path(img_path)[0]).shape
    json_dict = {}
    for p in range(len(convert_from_path(img_path))):
        json_dict[page_name(p)] = dict()
    return json_dict

def display_text(text, time):
    print(text)

def set_img_index(x):
    global img_index, pdf_page, img
    img_index = x
    pdf_page = 0
    img_path = IMAGE_PATH_LIST[img_index]
    img = cv2.cvtColor(np.array(convert_from_path(img_path)[pdf_page]), cv2.COLOR_BGR2RGB)
    text = 'Showing image {}/{}, path: {}'.format(str(img_index), str(last_img_index), img_path)
    display_text(text, 1000)

def set_page_index(x):
    global pdf_page, img
    pdf_page = x
    img_path = IMAGE_PATH_LIST[img_index]
    img = cv2.cvtColor(np.array(convert_from_path(img_path)[pdf_page]), cv2.COLOR_BGR2RGB)
    text = 'Showing image {}/{}, path: {}'.format(str(img_index), str(last_img_index), img_path)
    display_text(text, 1000)


def set_class_index(x):
    global class_index
    class_index = x
    text = 'Selected class {}/{} -> {}'.format(str(class_index), str(last_class_index), CLASS_LIST[class_index])
    display_text(text, 3000)


def draw_edges(tmp_img):
    blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
    edges = cv2.Canny(blur, 150, 250, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    tmp_img = np.bitwise_or(tmp_img, edges)
    #tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
    return tmp_img


def decrease_index(current_index, last_index):
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index


def increase_index(current_index, last_index):
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index


def draw_line(img, x, y, height, width, color):
    cv2.line(img, (x, 0), (x, height), color, LINE_THICKNESS)
    cv2.line(img, (0, y), (width, y), color, LINE_THICKNESS)

def findIndex(obj_to_find):
    #return [(ind, img_objects[ind].index(obj_to_find)) for ind in xrange(len(img_objects)) if item in img_objects[ind]]
    ind = -1

    ind_ = 0
    for listElem in img_objects:
        if listElem == obj_to_find:
            ind = ind_
            return ind
        ind_ = ind_+1

    return ind

def get_anchors_rectangles(xmin, ymin, xmax, ymax):
    anchor_list = {}

    mid_x = (xmin + xmax) / 2
    mid_y = (ymin + ymax) / 2

    L_ = [xmin - dragBBox.sRA, xmin + dragBBox.sRA]
    M_ = [mid_x - dragBBox.sRA, mid_x + dragBBox.sRA]
    R_ = [xmax - dragBBox.sRA, xmax + dragBBox.sRA]
    _T = [ymin - dragBBox.sRA, ymin + dragBBox.sRA]
    _M = [mid_y - dragBBox.sRA, mid_y + dragBBox.sRA]
    _B = [ymax - dragBBox.sRA, ymax + dragBBox.sRA]

    anchor_list['LT'] = [L_[0], _T[0], L_[1], _T[1]]
    anchor_list['MT'] = [M_[0], _T[0], M_[1], _T[1]]
    anchor_list['RT'] = [R_[0], _T[0], R_[1], _T[1]]
    anchor_list['LM'] = [L_[0], _M[0], L_[1], _M[1]]
    anchor_list['RM'] = [R_[0], _M[0], R_[1], _M[1]]
    anchor_list['LB'] = [L_[0], _B[0], L_[1], _B[1]]
    anchor_list['MB'] = [M_[0], _B[0], M_[1], _B[1]]
    anchor_list['RB'] = [R_[0], _B[0], R_[1], _B[1]]

    return anchor_list


def draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color):
    anchor_dict = get_anchors_rectangles(xmin, ymin, xmax, ymax)
    for anchor_key in anchor_dict:
        x1, y1, x2, y2 = anchor_dict[anchor_key]
        cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    return tmp_img

def draw_bboxes_from_file(tmp_img, annotation_path, width, height):
    global img_objects#, is_bbox_selected, selected_bbox
    img_objects = []
    if os.path.isfile(annotation_path):
        f = open(annotation_path)
        data = json.load(f)
        for bbox in data[page_name(pdf_page)].keys():
                xmin = data[page_name(pdf_page)][bbox]['xmin']
                xmax = data[page_name(pdf_page)][bbox]['xmax']
                ymin = data[page_name(pdf_page)][bbox]['ymin']
                ymax = data[page_name(pdf_page)][bbox]['ymax']
                # draw bbox
                cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)
                # draw resizing anchors if the object is selected
                if is_bbox_selected:
                    if idx == selected_bbox:
                        tmp_img = draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(tmp_img, bbox, (xmin, ymin - 5), font, 0.6, color, LINE_THICKNESS, cv2.LINE_AA)

    return tmp_img

def delete_bbox_from_file(img_path, annotation_path, del_class):
    if del_class == 'none':
        return

    if os.path.isfile(annotation_path):
        f = open(annotation_path)
        data = json.load(f)
        if del_class == '*':
            data = empty_json(img_path)
        else:
            data[page_name(pdf_page)].pop(del_class, None)

        with open(annotation_path, 'w') as f:
            json.dump(data, f)

def get_bbox_area(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width*height


def set_selected_bbox(set_class):
    global is_bbox_selected, selected_bbox
    smallest_area = -1
    # if clicked inside multiple bboxes selects the smallest one
    for idx, obj in enumerate(img_objects):
        ind, x1, y1, x2, y2 = obj
        x1 = x1 - dragBBox.sRA
        y1 = y1 - dragBBox.sRA
        x2 = x2 + dragBBox.sRA
        y2 = y2 + dragBBox.sRA
        if pointInRect(mouse_x, mouse_y, x1, y1, x2, y2):
            is_bbox_selected = True
            tmp_area = get_bbox_area(x1, y1, x2, y2)
            if tmp_area < smallest_area or smallest_area == -1:
                smallest_area = tmp_area
                selected_bbox = idx
                if set_class:
                    # set class to the one of the selected bounding box
                    cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, ind)


def is_mouse_inside_delete_button():
    for idx, obj in enumerate(img_objects):
        if idx == selected_bbox:
            _ind, x1, y1, x2, y2 = obj
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            if pointInRect(mouse_x, mouse_y, x1_c, y1_c, x2_c, y2_c):
                return True
    return False


def edit_bbox(obj_to_edit, action):
    ''' action = `delete`
                 `change_class:new_class_index`
                 `resize_bbox:new_x_left:new_y_top:new_x_right:new_y_bottom`
    '''
    if 'change_class' in action:
        new_class_index = int(action.split(':')[1])
    elif 'resize_bbox' in action:
        new_x_left = max(0, int(action.split(':')[1]))
        new_y_top = max(0, int(action.split(':')[2]))
        new_x_right = min(width, int(action.split(':')[3]))
        new_y_bottom = min(height, int(action.split(':')[4]))

    # 1. initialize bboxes_to_edit_dict
    #    (we use a dict since a single label can be associated with multiple ones in videos)
    bboxes_to_edit_dict = {}
    current_img_path = IMAGE_PATH_LIST[img_index]
    bboxes_to_edit_dict[current_img_path] = obj_to_edit

def mouse_listener(event, x, y, flags, param):
    # mouse callback function
    global is_bbox_selected, prev_was_double_click, mouse_x, mouse_y, point_1, point_2

    set_class = True
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        prev_was_double_click = True
        #print('Double click')
        point_1 = (-1, -1)
        # if clicked inside a bounding box we set that bbox
        set_selected_bbox(set_class)
    # By AlexeyGy: delete via right-click
    elif event == cv2.EVENT_RBUTTONDOWN:
        set_class = False
        set_selected_bbox(set_class)
        if is_bbox_selected:
            obj_to_edit = img_objects[selected_bbox]
            edit_bbox(obj_to_edit, 'delete')
            is_bbox_selected = False
    elif event == cv2.EVENT_LBUTTONDOWN:
        if prev_was_double_click:
            #print('Finish double click')
            prev_was_double_click = False
        else:
            #print('Normal left click')

            # Check if mouse inside on of resizing anchors of the selected bbox
            if is_bbox_selected:
                dragBBox.handler_left_mouse_down(x, y, img_objects[selected_bbox])

            if dragBBox.anchor_being_dragged is None:
                if point_1[0] == -1:
                    if is_bbox_selected:
                        if is_mouse_inside_delete_button():
                            set_selected_bbox(set_class)
                            obj_to_edit = img_objects[selected_bbox]
                            edit_bbox(obj_to_edit, 'delete')
                        is_bbox_selected = False
                    else:
                        # first click (start drawing a bounding box or delete an item)

                        point_1 = (x, y)
                else:
                    # minimal size for bounding box to avoid errors
                    threshold = 5
                    if abs(x - point_1[0]) > threshold or abs(y - point_1[1]) > threshold:
                        # second click
                        point_2 = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if dragBBox.anchor_being_dragged is not None:
            dragBBox.handler_left_mouse_up(x, y)



def get_close_icon(x1, y1, x2, y2):
    percentage = 0.05
    height = -1
    while height < 15 and percentage < 1.0:
        height = int((y2 - y1) * percentage)
        percentage += 0.1
    return (x2 - height), y1, x2, (y1 + height)


def draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c):
    red = (0,0,255)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), red, -1)
    white = (255, 255, 255)
    cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
    return tmp_img


def draw_info_bb_selected(tmp_img):
    for idx, obj in enumerate(img_objects):
        ind, x1, y1, x2, y2 = obj
        if idx == selected_bbox:
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
    return tmp_img


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def convert_video_to_images(video_path, n_frames, desired_img_format):
    # create folder to store images (if video was not converted to images already)
    file_path, file_extension = os.path.splitext(video_path)
    # append extension to avoid collision of videos with same name
    # e.g.: `video.mp4`, `video.avi` -> `video_mp4/`, `video_avi/`
    file_extension = file_extension.replace('.', '_')
    file_path += file_extension
    video_name_ext = os.path.basename(file_path)
    if not os.path.exists(file_path):
        print(' Converting video to individual frames...')
        cap = cv2.VideoCapture(video_path)
        os.makedirs(file_path)
        # read the video
        for i in tqdm(range(n_frames)):
            if not cap.isOpened():
                break
            # capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # save each frame (we use this format to avoid repetitions)
                frame_name =  '{}_{}{}'.format(video_name_ext, i, desired_img_format)
                frame_path = os.path.join(file_path, frame_name)
                cv2.imwrite(frame_path, frame)
        # release the video capture object
        cap.release()
    return file_path, video_name_ext


def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def get_annotation_path(img_path):
    new_path = os.path.join(OUTPUT_DIR, os.path.basename(os.path.normpath(img_path)))
    pre_path, img_ext = os.path.splitext(new_path)
    annotation_path = new_path.replace(img_ext, '.json', 1)
    return annotation_path

def save_bounding_box(img_path, annotation_path, class_index, point_1, point_2, width, height):
    xmin = min((point_1[0], point_2[0]))
    xmax = max((point_1[0], point_2[0]))
    ymin = min((point_1[1], point_2[1]))
    ymax = max((point_1[1], point_2[1]))

    if os.path.isfile(annotation_path):
        f = open(annotation_path)
        data = json.load(f)
    else:
        data = empty_json(img_path)

    class_name = input('Class Name: ')
    type_name = input('Type Name: ')
    divs = 0
    if 'text' in type_name:
        divs = int(input('Divs: '))
    data[page_name(pdf_page)][class_name] = { 'type':type_name,
                                  'divs':divs,
                                  'xmin':xmin,
                                  'xmax':xmax,
                                  'ymin':ymin,
                                  'ymax':ymax }

    with open(annotation_path, 'w') as f:
        json.dump(data, f)

def complement_bgr(color):
    lo = min(color)
    hi = max(color)
    k = lo + hi
    return tuple(k - u for u in color)

# change to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))



if __name__ == '__main__':
    # load all images and videos (with multiple extensions) from a directory using OpenCV
    IMAGE_PATH_LIST = []
    for f in sorted(os.listdir(INPUT_DIR), key = natural_sort_key):
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

    last_img_index = len(IMAGE_PATH_LIST) - 1

    print(IMAGE_PATH_LIST)

    # create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1000, 700)
    cv2.setMouseCallback(WINDOW_NAME, mouse_listener)

    # initialize
    set_img_index(0)
    edges_on = False

    move = False

    display_text('Welcome!\n Press [h] for help.', 4000)

    # loop
    while True:
        #color = class_rgb[class_index].tolist()
        # clone the img
        tmp_img = img.copy()
        height, width = tmp_img.shape[:2]
        if edges_on == True:
            # draw edges
            tmp_img = draw_edges(tmp_img)
        # draw vertical and horizontal guide lines
        draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
        # write selected class
        #class_name = CLASS_LIST[class_index]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        margin = 3

        img_path = IMAGE_PATH_LIST[img_index]

        annotation_path = get_annotation_path(img_path)
        if dragBBox.anchor_being_dragged is not None:
            dragBBox.handler_mouse_move(mouse_x, mouse_y)
        # draw already done bounding boxes
        tmp_img = draw_bboxes_from_file(tmp_img, annotation_path, width, height)
        # if bounding box is selected add extra info
        if is_bbox_selected:
            pass
            #tmp_img = draw_info_bb_selected(tmp_img)
        # if first click
        if point_1[0] != -1 and not move:
            # draw partial bbox
            cv2.rectangle(tmp_img, point_1, (mouse_x, mouse_y), color, LINE_THICKNESS)
            # if second click
            if point_2[0] != -1 and not move:
                # save the bounding box
                save_bounding_box(img_path, annotation_path, class_index, point_1, point_2, width, height)
                # reset the points
                point_1 = (-1, -1)
                point_2 = (-1, -1)

        cv2.imshow(WINDOW_NAME, tmp_img)
        pressed_key = cv2.waitKey(DELAY)

        if dragBBox.anchor_being_dragged is None:
            ''' Key Listeners START '''
            if pressed_key == ord('m'):
                # reset the points
                point_1 = (-1, -1)
                point_2 = (-1, -1)
                if move:
                    move = False
                    display_text('Move mode disabled', 1000)
                else:
                    move = True
                    display_text('Move mode enabled', 1000)

            if pressed_key == ord('a') or pressed_key == ord('d'):
                last_pdf_index = len(convert_from_path(img_path)) - 1
                # show previous image key listener
                if pressed_key == ord('a'):
                    page_index = decrease_index(pdf_page, last_pdf_index)
                # show next image key listener
                elif pressed_key == ord('d'):
                    page_index = increase_index(pdf_page, last_pdf_index)
                set_page_index(page_index)

            if pressed_key == ord('z') or pressed_key == ord('x'):
                # show previous image key listener
                if pressed_key == ord('z'):
                    img_index = decrease_index(img_index, last_img_index)
                # show next image key listener
                elif pressed_key == ord('x'):
                    img_index = increase_index(img_index, last_img_index)
                set_img_index(img_index)

            elif pressed_key == ord('h'):
                text = ('[e] to show edges;\n'
                        '[q] to quit;\n'
                        '[a] or [d] to change page within pdf;\n'
                        '[z] or [x] to change pdf document;\n'
                        '[m] to enable / disable box drawing;\n'
                        '[c] to remove a field by name;\n'
                        )
                display_text(text, 5000)
            # show edges key listener
            elif pressed_key == ord('e'):
                if edges_on == True:
                    edges_on = False
                    display_text('Edges turned OFF!', 1000)
                else:
                    edges_on = True
                    display_text('Edges turned ON!', 1000)

            elif pressed_key == ord('c'):
                display_text('Enter a field name to delete ("none" cancels and "*" deletes all fields):', 1000)
                del_class = input()
                delete_bbox_from_file(img_path, annotation_path, del_class)

            elif pressed_key == ord('q'):
                break
            ''' Key Listeners END '''

        if WITH_QT:
            # if window gets closed then quit
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
