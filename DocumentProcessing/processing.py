import cv2
import numpy as np
from math import floor, ceil

import SimpleHTR
import WordSegmentation

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def print_boxes(img, word_boxes):
	# Create figure and axes
	fig,ax = plt.subplots(1)

	# Display the image
	ax.imshow(img)

	for w in word_boxes:
	    x, y, w, h = w
	    
	    rect = patches.Rectangle((x, y),w,h,linewidth=1,edgecolor='r',facecolor='none')
	    ax.add_patch(rect)
	    
	plt.show()

# preprocess image
def preprocess_img(img):
	# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=adaptivethreshold
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 10)
	return img

def shrink_img(img, boxes):
	if len(boxes)==0:
		return np.ones((10, 10))*254, boxes
	else:
		miny = np.min(boxes[:,1])
		maxy = min(np.max(boxes[:,1])+np.max(boxes[:,3]), img.shape[0])
		minx = np.min(boxes[:,0])
		maxx = min(np.max(boxes[:,0])+np.max(boxes[:,2]), img.shape[1])

		new_img = img[miny:maxy, minx:maxx]
		new_boxes = boxes[:]
		new_boxes[:,0] = new_boxes[:,0]-minx
		new_boxes[:,1] = new_boxes[:,1]-miny

		return new_img, new_boxes



def remove_empty_boxes(img, boxes, threshold=0.05):
	full_boxes = []
	for box in boxes:
		value = np.mean(img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])
		if (value/255)<=(1-threshold):
			full_boxes.append(box)
	return np.array(full_boxes)

# remove redundant boxes from array
def remove_subboxes(old_boxes):
	boxes = []
	for b1 in old_boxes:
		is_subbox = False
		for b2 in old_boxes:
			#print(b1, b2)
			if (b1[0]>=b2[0]) and (b1[1]>=b2[1]) and (b1[0]+b1[2]<=b2[0]+b2[2]) and \
				(b1[1]+b1[3]<=b2[1]+b2[3]) and not (b1==b2):
				is_subbox=True
		if not is_subbox:
			boxes.append(b1)
	return np.array(boxes)

# get next line of words. Used in wort word boxes
def next_line(words, avg_height, height_threshold):
	topy = np.min(words[:,1])
	threshold = topy + (avg_height*height_threshold)   
	line = []
	remainder = []
	for word in words:
		if word[1] <= threshold:
			line.append(word)
		else:
			remainder.append(word)	            
	return np.array(line), np.array(remainder)

# return sorted list or word boxes
def sort_word_boxes(word_boxes, height_threshold=0.85):
	if not len(word_boxes):
		return word_boxes
	avg_height = np.mean(word_boxes[:,3])
	sorted_words = []
	while word_boxes.shape[0]:
		line, word_boxes = next_line(word_boxes, avg_height, height_threshold)
		sorted_line = sorted(list(line), key=lambda x:x[0])
		sorted_words = sorted_words+sorted_line
	return np.array(sorted_words)

def segment_word_box(word_box, wh_ratio=15):
	box_ratio = word_box[2]/word_box[3]

	segments = []
	for i in range(floor(box_ratio/wh_ratio)):
		newx = word_box[0]+(i*word_box[3]*wh_ratio)
		newy = word_box[1]
		neww = word_box[3]*wh_ratio
		newh = word_box[3]
		segments.append(np.array([newx, newy, neww, newh]))
	lastx = word_box[0]+(floor(box_ratio/wh_ratio)*word_box[3]*wh_ratio)
	lasty = word_box[1]
	lastw = word_box[2]-(floor(box_ratio/wh_ratio)*word_box[3]*wh_ratio)
	lasth = word_box[3]
	segments.append(np.array([lastx, lasty, lastw, lasth]))

	print(segments)

	return segments

def reset_session():
	SimpleHTR.reset_session()

def init_model():
	global model
	model = SimpleHTR.default_model(charList='htr_model/charList.txt', modelDir='htr_model/')

def process_text_field(field):
	field = preprocess_img(field)

	trans = []
	prob = []
	for sub_box in segment_word_box([0, 0, field.shape[1], field.shape[0]], 15):
		x = sub_box[0]
		y = sub_box[1]
		w = sub_box[2]
		h = sub_box[3]

		t, p = SimpleHTR.infer_img(model, field[y:y+h, x:x+w])
		trans.append(t)
		prob.append(p)

	transcription = ''.join(trans)
	proba = np.mean(prob)
	return transcription, proba

# return the contents and confidence of a text multiline field
def process_large_text_field(field):
	field = preprocess_img(field)
	word_boxes = WordSegmentation.wordSegmentation(field, kernelSize=15, sigma=10, theta=20, minArea=500)
	word_boxes = [x[0] for x in word_boxes]
	word_boxes = remove_subboxes(word_boxes)
	word_boxes = remove_empty_boxes(field, word_boxes)
	field, word_boxes = shrink_img(field, word_boxes)
	word_boxes = sort_word_boxes(word_boxes)

	print_boxes(field, word_boxes)

	transcription = []
	probas = []
	#model = SimpleHTR.default_model(charList='htr_model/charList.txt', modelDir='htr_model/')
	for word_box in word_boxes:

		trans = []
		prob = []
		for sub_box in segment_word_box(word_box, 15):
			x = sub_box[0]
			y = sub_box[1]
			w = sub_box[2]
			h = sub_box[3]

			t, p = SimpleHTR.infer_img(model, field[y:y+h, x:x+w])
			trans.append(t)
			prob.append(p)

		transcription.append(''.join(trans))
		probas.append(np.mean(prob))	

	#SimpleHTR.reset_session()
	if len(word_boxes):
		transcription_text = ' '.join(transcription)
		mean_proba = np.mean(np.array(probas))
	else:
		transcription_text = ''
		mean_proba = None

	return transcription_text, mean_proba

# return the contents and confidence of a checkbox
def process_checkbox(field):
	field = preprocess_img(field)
	value = np.mean(field) / 255

	threshold = 0.6
	transcription = (value<=threshold)
	proba = (1-threshold) + abs(value - threshold)

	return transcription, proba

if __name__=='__main__':
	sample_text_field = cv2.cvtColor(cv2.imread("sample_forms/gucci.jpg"), cv2.COLOR_BGR2GRAY)
	trans, proba = process_text_field(sample_text_field)
	print('Transcribed text: ', trans)
	print('Transcription confidence: ', proba)

	trans, proba = process_checkbox(sample_text_field)
	print('Checkbox value: ', trans)
	print('Checkbox confidence: ', proba)