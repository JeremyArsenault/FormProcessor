## Tool for processing PDF forms with handwitten fields

### Quick Start:

Put pdf files of forms with handwritten fields to be labeled in the input directory. Corresponding .json files containing processed file information will be created in output directory.

Note: You must first create corresponding 'pdf_file_name.json' layout file before processing

$ main.py [-h] [-i] [-o] [-t]

	required arguments:
	-l, --layout			   Path to layout.json file
	optional arguments:
	 -h, --help                Show this help message and exit
	 -i, --input               Path to pdf input folder | Default: input/
	 -o, --output              Path to pdf folder | Default: output/

### Layout Format:

	{
	page-n : {
		field-name : {
			type : str // ('text-alpha', 'text-num', 'text-alphanum', 'checkbox')
			value : str or bool
			proba : str(float)
		}
	}