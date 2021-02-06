import os
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import pandas as pd


# Defining directories
root_dir = os.getcwd()
annotations_dir = os.path.join(root_dir, 'annotations')
images_dir = os.path.join(root_dir, 'images')

# Sorting images and labels to be ordered the same
images_list = list(sorted(os.listdir(images_dir)))
labels_list = list(sorted(os.listdir(annotations_dir)))

def get_targets(annotation):
    '''
    get_targets function uses the annotation url argument to read through the xml file in order to
    return a list of bounding boxes ([xmin, ymin, xmax, ymax]) and a list of labels
    :param annotation -> url
    :return:bbox_list ,labels_list
    '''
    with open(annotation) as f:
        label_dict = {'without_mask': 0, 'with_mask': 1, 'mask_weared_incorrect': 2}
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')
        label_l = []
        bbox_l = []
        for obj in objects:
            xmin = int(obj.find('xmin').text)
            ymin = int(obj.find('ymin').text)
            xmax = int(obj.find('xmax').text)
            ymax = int(obj.find('ymax').text)
            label = label_dict[obj.find('name').text]
            label_l.append(label)
            bbox_l.append([xmin,ymin,xmax,ymax])
    return bbox_l, label_l


def crop_image(image,bbox_list):
    '''
    take as input an image and the corresponding list of bounding boxes and returns
    a list of cropped images
    :param image -> url
    :param bbox_list:
    :return:list of cropped images
    '''
    image = np.array(Image.open(image))
    images_list = []
    for bbox in bbox_list:
        xmin, ymin, xmax, ymax = bbox
        img_copy = image.copy()
        img_copy = img_copy[ymin:ymax+1, xmin:xmax+1]
        images_list.append(img_copy)

    return images_list

# Changing directory to save in "cropped images" directory
cropped_images_dir = os.path.join(root_dir, 'cropped images')
os.chdir(cropped_images_dir)


# The following code does:
# 1.Gets url of annotation and image
# 2.Gets the bounding boxes and labels from the get_targets function
# 3.Crops the image to multiple images
# 4.Appending the labels to labels_values_list
# 5.Saving the image

labels_values_list = []
cropped_images_name_list = []
i = 1
for image, label in zip(images_list,labels_list):
    annot_url = os.path.join(annotations_dir,label)
    img_url = os.path.join(images_dir, image)
    bounding_boxes, labels = get_targets(annot_url)
    cropped_image_list = crop_image(img_url, bounding_boxes)
    labels_values_list.append(labels)

    for cropped_image in cropped_image_list:
        img = Image.fromarray(cropped_image)
        name = 'mask' + str(i) + '.png'
        img.save(name)
        cropped_images_name_list.append(name)
        i += 1


# labels_values_list is a list of lists. unpack_nested_lists unpacks the nested lists inside the labels_values_list
unpacked_labels = []
def unpack_nested_lists(l):
    for item in l:
        if isinstance(item, list):
            unpack_nested_lists(item)
        else:
            unpacked_labels.append(item)

unpack_nested_lists(labels_values_list)

df = pd.DataFrame(data=[cropped_images_name_list, unpacked_labels])
df = df.transpose()
df.columns = ['image', 'label']
# Saving the labels to a csv file
os.chdir(root_dir)
df.to_csv('labels.csv',index=False)