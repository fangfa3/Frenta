import glob
import json
import os

filenames = glob.glob('..\\..\\12k\\labels_12k\\*')
output_path = '12k_labels.txt'
classes = ["door","robot","football"]
with open(output_path,'w') as fo:
	for filename in filenames:
		imagepath = '12k\\images_12k\\'+os.path.basename(filename)[:-5]
		annotation = imagepath
		with open(filename) as f:
			df = json.load(f)
			if 'Rects' in df:
				for obj in df['Rects']:
					x_min = int(obj['x'])
					y_min = int(obj['y'])
					x_max = int(obj['x'] + obj['w'])
					y_max = int(obj['y'] + obj['h'])
					cls = obj['properties']['world_cup'][0]
					class_id = classes.index(cls) 
					annotation+=' '+','.join(str(num) for num in [x_min,y_min,x_max,y_max,class_id])
		# print(annotation)
		fo.write(annotation+'\n')
