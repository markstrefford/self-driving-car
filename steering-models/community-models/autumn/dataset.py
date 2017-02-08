# Read through processed files and create a output csv with the remaining files in the directory
data_dir = '/vol/data'
csv_file = './dataset.txt'
csv_out = './remaining.txt'

import os
import csv

print 'Reading image dir...'
all_images = os.listdir(data_dir)
print 'Found {} total images'.format(len(all_images))

print 'Loading processed image list...'
with open(csv_file, 'rb') as f:
    reader = csv.reader(f)
    processed_images = list(reader)
print 'Found {} processed images'.format(len(processed_images))

print 'Identifying unprocessed images...'

remaining_images = [image for image in all_images if ([image] not in processed_images and image.endswith('.png'))]
print 'Identified {} unprocessed images'.format(len(remaining_images))
for f in range(len(remaining_images)):
    print remaining_images[f]


