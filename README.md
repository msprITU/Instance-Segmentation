# Instance Segmentation
<!-- # MSPR ITU Dataset - Face Segmentation --->
We constructed a dataset for instance segmentation tasks. We segmented Pascal VOC 2012 dataset. While there are 1454 images and 2695 face instances, we applied filtering to omit some outliers, and our dataset contains 2667 face instances with their localization and segmentation information. In the following figures, you may observe a frame and segmentation of face object.


| ![Image](siu2019/Images/000001.jpg)  |  ![Segmentation](siu2019/Masks_GT/000001_mask_0.png) |
| --- | --- |

The dataset can be accessed through [here](/siu2019). 

In the directory, there are three folders which are [Annotation](processed-Data/Annotations/), [Images](processed-Data/Images/), [Masks](/siu2019/Masks) and [Masks_GT](/siu2019/Masks_GT) which contain annotation information in XML format, images in jpg format, segmentation masks before and after filtering, respectively. However, in our model we have created such dictionary for making it possible to store the segmentation, area, category_id, image_id, bbox and iscrowd parameters in JSON format, in order to train Mask R-CNN.

### Parameters in JSON file

    segmentation: the encoded binary mask information. 
                  (list) of integers > [217,48,216,49,215,50,214,51,213,...,50]        
    area        : total number of pixels belong to the instance. 
                  (int) > 19918
    category_id : id of object. 
                  (str) > "face"
    image_id    : id of image. 
                  (int) > 600000
    bbox        : location of instance. 
                  (list) of int > [138, 48, 142, 173]
    iscrowd     : if the frame contains one or more instances. 
                  (bool) > 0
                  
### Creating Microsoft COCO-like Dataset for Mask R-CNN

We created our dataset by using [LabelMe Annotation Tool](http://labelme.csail.mit.edu/Release3.0/). LabelMe Annotation Tool is a semi-automatic segmentation tool can be used through a browser. The method performs scene recognition at backend. Even though there are many tools for semi-automatic segmentation, we utilized LabelMe since we concluded that it has a high performance in efficient time cost.

### Requirements
```
pip install json, cv2, xml, skimage, pycocotools, scipy, skimate, skimage, urlib
```
We will explain how to create a Microsoft COCO-like dataset and how to train Mask R-CNN with a custom dataset. Firstly, instances should be annotated with the tool. 

We wrote a [tool](Routine/dataset_creator.py) for converting LabelMe outputs into proper representation for training Mask R-CNN. After downloading all the outputs, we stored as in [./source](/Routine/Source) directory. There should be 3 folders, namely, [/Annotations](/Routine/Source/Annotations), [/Images](/Routine/Source/Images) and [/Masks](/Routine/Source/Masks) folders. After this preparation, it is possible to run the routine by

```
python3 dataset_creator.py --source [source] --destination [destination] --objective [objective] --objects [objects]
```
where;

* "source" argument is the directory of LabelMe Annotation Tool outputs.
* "destination" argument is the directory of the routine's outputs.
* "objective" argument is selection of usage of data, either 'train' or 'val'
* "objects" argument is a string of objects splitted with a comma without any spacing: 'obj1,obj2,obj3'

### Functionality of Routine

Our method reads through .xml annotations and extracts parameters in the following: **"deleted_flags, polygon, bbox, iscrowd"** and mask filenames. 

* **"deleted_flags"** is a parameter that indicates if the instance is deleted after being annotated. It is possible that the annotator would like to improve the output by deleting the instance that the annotator created and reannotating it again. In this case, LabelMe still keeps deleted instances. By extracting deleted_flags parameter, we can eliminate deleted instances. 

* **"polygon"** is a parameter that indicates if the instance is annotated with polygon representation. Since our method currently does not support polygon annotations, we skip the instances that are annotated using polygons. 

* **"bbox"** is the parameter that indicates the location of instance. While LabelMe stores the location as [x1,y1,x2,y2], it is converted into [x1,y1,w,h].

* **"iscrowd"** is the parameter that indicates if the image contains a single instance or multiple instances. It is a boolean value which is extracted from the .xml files.

After these operations, the routine extracts mask filenames and reads each binary images from "Masks_GT" directory.

The tool reads each instance, thresholds it in order to represent the mask with zeros and ones (e.g., black and white pixels). 

It is observed that the output of the tool has some noisy pixels which are undesirable. The noisy pixels can be either black dots inside the instance or white dots outside the instance. In this case, holes are filled with white dots and small objects are removed. Lastly, we use RLE Encoding in order to compress .png mask into list of integers.
