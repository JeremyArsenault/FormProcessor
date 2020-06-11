## Tool for labeling fields on documents

Adapted from OpenLabeler https://github.com/Cartucho/OpenLabeling

### Quick Start:

Put pdf files of documents to be labeled in the input directory

Corresponding 'pdf_file_name.json' layout file will be created in output directory

$ main.py [-h] [-i] [-o] [-t]

	optional arguments:
	 -h, --help                Show this help message and exit
	 -i, --input               Path to images input folder | Default: input/
	 -o, --output              Path to output folder | Default: output/
	 -t, --thickness           Bounding box and cross line thickness (int) | Default: -t 3

### Usage:

Left click once to begin drawing a box, and again to finish. Then enter field name in terminal. Press [m] to stop drawing a box. Pinch / 'ctrl' + scroll to zoom, click and drag to move window. Use [a] and [d] to switch between images and [q] to quit.

	[e] to show edges
    [q] to quit
    [a] or [d] to change page
    [z] or [x] to change pdf
    [m] to enable / disable box drawing
    [c] to remove a field by name