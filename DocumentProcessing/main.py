import argparse


if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Open-source document process tool')
	parser.add_argument('-i', '--input_dir', default='input', type=str, help='Path to input directory')
	parser.add_argument('-o', '--output_dir', default='output', type=str, help='Path to output directory')
	parser.add_argument('-m', '--htr_dir', default='htr_model', type=str, help='Path to htr model.py directory')
	parser.add_argument('-c', '--cbox_dir', default='checkbox_model', type=str, help='Path to checkbox model.py directory')

	args = parser.parse_args()