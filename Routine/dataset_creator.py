import json
import numpy as np
import skimage.io as io
import os
from pycocotools import coco
import xml.etree.ElementTree as ET
import xml
from pycocotools import mask as comask
import cv2
from utils import DateCaptured, coco_bbox_creator, iou_calculator, mask2poly,binary_mask_to_rle,isjpg,ispng,save_bboxes,xml_retreiver

import datetime
import itertools


json_structure={'annotations': [],
 'categories': [],
 'images': [],
 'info': {'contributor': 'MSPR ITU',
  'date_created': '2019-01-16 11:30:52.357475',
  'description': 'Microsoft COCO Style Custom Dataset.',
  'url': 'https://github.com/mspritu',
  'version': '1.0',
  'year': 2019}}





if __name__=='__main__':
    import argparse
    parser= argparse.ArgumentParser("Dataset Creation Routine.")
    parser.add_argument('--source',required=True, metavar='path/to/labelmeoutput')
    parser.add_argument('--objects',required=True,help="Comma separated.")
    parser.add_argument('--destination',required=True)
    parser.add_argument('--obj',required=True)


    args = parser.parse_args()
    main_dataset_dir=args.source
    objects=args.objects.split(',')
    dest_name=args.destination
    objective=args.obj
    if not os.path.exists(dest_name):
        os.makedirs(os.path.join(dest_name,"Images"))
        os.makedirs(os.path.join(dest_name,'processed_masks'))
        os.makedirs(os.path.join(dest_name,'Annotations'))
    dict_of_categories={91+indx:name for indx,name in enumerate(objects)}
    for indx,name in enumerate(objects):
        json_structure['categories'].append({'id':91+indx,'name':name,'supercategory':'object'})
    image_id = 600000
    annotation_id = 600000
    for class_name in dict_of_categories.keys():
        annot_dir = os.path.join(main_dataset_dir, "Annotations/")
        mask_dir = os.path.join(main_dataset_dir, "Masks/")
        image_dir = os.path.join(main_dataset_dir, "Images/") 

        annot_all_files = sorted(os.listdir(annot_dir))

        for xml_file in annot_all_files:

            tree = ET.parse(os.path.join(annot_dir, xml_file))
            root = tree.getroot()

            delete_flags = root.findall("./object/deleted")
            sum_deleted = sum(map(int, [delete_flag.text for delete_flag in delete_flags]))

            xmins = root.findall("./object/segm/box/xmin")
            ymins = root.findall("./object/segm/box/ymin")
            xmaxs = root.findall("./object/segm/box/xmax")
            ymaxs = root.findall("./object/segm/box/ymax")

            mask_filenames = root.findall("./object/segm/mask")

            # Not yet polygon type of annotations are supported.
            assert root.findall("./object/polygon") == [], "Polygon type of annotations are not yet to be supported."

            # IMAGES
            # Image will be copied to MS COCO train directory and file name will be hold

            image_filename = xml_retreiver(root, 'filename')

            assert image_filename[-4:] == ".jpg", "The filename extension is not .jpg. Name of file: {}".format(image_filename)
            image = io.imread(os.path.join(image_dir, image_filename))

            if objective == "train":
                if image_id<1000000:
                    image_target_filename = "MSPRtrain2014_000000"+str(image_id)+".jpg"
                elif (image_id>=1000000 or image_id<=10000000):
                    image_target_filename = "MSPRtrain2014_00000"+str(image_id)+".jpg"
            elif objective == "valid":
                if image_id<1000000:
                    image_target_filename = "MSPRval2014_000000"+str(image_id)+".jpg"
                elif (image_id>=1000000 or image_id<=10000000):
                    image_target_filename = "MSPRval2014_00000"+str(image_id)+".jpg"

    
            io.imsave(os.path.join(dest_name,"Images", image_target_filename), image)

            # Image height and width
            height = int(xml_retreiver(root, 'nrows'))
            width = int(xml_retreiver(root, 'ncols'))

            assert height == image.shape[0]
            assert width == image.shape[1]


            # Miscellaneous metadata
            date_captured = DateCaptured()
            coco_url = 'http://www.mspr.itu.edu.tr/'
            flickr_url = 'http://www.mspr.itu.edu.tr/'
            license = np.random.randint(8)

            # Appending to 'images'
            json_structure['images'].append({'coco_url': coco_url, 'file_name': image_target_filename, 
                                        'date_captured': date_captured, 'flickr_url': flickr_url,
                                        'height': height, 'id': image_id, 'license': license, 
                                        'width': width})
            # Polygon Handling

            for d, delete_flag in enumerate(delete_flags):
                delete_flag = int(delete_flag.text)
                if not delete_flag:

                    # ANNOTATIONS
                    # Bbox
                    bbox_x = int(xmins[d].text)
                    bbox_y = int(ymins[d].text)
                    bbox_w = int(xmaxs[d].text) - bbox_x
                    bbox_h = int(ymaxs[d].text) - bbox_y
                    bbox_list = [bbox_x, bbox_y, bbox_w, bbox_h]
                    save_bboxes(image_filename[:-4], bbox_list)

                    # Class
                    category_id = dict_of_categories[class_name]

                    # Masks Segmentation
                    mask = cv2.imread(os.path.join(mask_dir, mask_filenames[d].text), 0)
                    polygons, binary_image = mask2poly((mask))

                    # Mask should be defined with at least 60 points.
                    area_binary_img = len(np.where(binary_image==1)[0])
                    #if area_binary_img < area_thr:
                        #continue

                    # Write the new modified masks to a folder
                    cv2.imwrite(os.path.join(dest_name,'processed_masks' ,mask_filenames[d].text), binary_image*255)

                    # Generate the RLE version of mask
                    RLE_mask = binary_mask_to_rle(mask)
                    RLE_mask_coco = comask.encode(np.asfortranarray(binary_image.astype(np.uint8)))

                    # Area
                    area = comask.area(RLE_mask_coco)

                    #  Iscrowd
                    if (len(mask_filenames) - sum_deleted) > 1:
                        iscrowd = 1
                        json_structure['annotations'].append({'iscrowd': iscrowd, 'bbox': bbox_list, 'id': annotation_id,
                                                        'image_id': image_id, 'segmentation': RLE_mask,
                                                        'area': area, 'category_id': dict_of_categories[class_name]})
                    else:
                        iscrowd = 0
                        json_structure['annotations'].append({'iscrowd': iscrowd, 'bbox': bbox_list, 'id': annotation_id,
                                                    'image_id': image_id, 'segmentation': polygons,
                                                    'area': area, 'category_id': dict_of_categories[class_name]})

                    # Appending into annotations

                    annotation_id+=1
            image_id+=1
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

with open(os.path.join(dest_name,"Annotations",'instances_train_2014.json'),"w") as f:
    data = json.dumps(json_structure, cls=MyEncoder, indent=4)
    f.write(data)


    





