import xml.etree.ElementTree as ET
import random
import glob

all_class = ['straw hat' , 'train' , 'person' , 'hat' , 'head' , 'cellphone' , 'cell phone']

act_class = ['hat' , 'head' , 'cellphone' , 'person' , 'lamp']
def Read_Write_classestxt_trainanno_(xml_root, img_root):


    xml_list = glob.glob(f'{xml_root}/*')
    print(xml_root)
    # random.seed(1)
    # random.shuffle(xml_list)
    annotation_lines = []
    classes = []
    for i, xml_name in enumerate(xml_list):
        # try:

        tree = ET.parse(xml_name)
        # root = tree.getroot()
        img_file_path = tree.findtext('./filename')
        obj_5features = ''
        for obj in tree.iter('object'):
            gt = obj.findtext('name')
            if gt == None:
                continue

            annotation_line = img_file_path + ' '
            for box in obj.iter('bndbox'):
                xmin = str(box.findtext('xmin')) + ','
                ymin = str(box.findtext('ymin')) + ','
                xmax = str(box.findtext('xmax')) + ','
                ymax = str(box.findtext('ymax')) + ','
                coor = xmin  + ymin + xmax + ymax + str(act_class.index(gt)) + ' '
                obj_5features += coor

            annotation_line += obj_5features
        annotation_lines.append(img_root + '/' + annotation_line)
        # exit()
        # except:
        #     # print(f'{img_root}/{img_file_path} hasn\'t box ')
        #     pass
    set_class = set(classes)

    # with open("classes.txt", 'w') as f:  # 可以传全局变量进来
    #     for a in act_class:
    #         f.write(a + '\n')
    return annotation_lines
    # with open("train_anno.txt", 'w') as f:  # 可以传全局变量进来
    #     for a in train_img_paths:
    #         f.write(img_root + '/' + a + '\n')


if __name__=="__main__":
    xml_roots = ['new_label_img/label']
    img_roots = ['new_label_img/image']

    base_anno_lines = []
    for xml_root,img_root in zip(xml_roots,img_roots):
        single_folder_anno_lines = Read_Write_classestxt_trainanno_(xml_root, img_root)
        print(len(single_folder_anno_lines))
        base_anno_lines.append(single_folder_anno_lines)

    with open("train_anno.txt", 'w') as f:  # 可以传全局变量进来
        for single_folder_anno_lines in base_anno_lines:
            for anno_line in single_folder_anno_lines:
                # print(anno_line)
                f.write(anno_line + '\n')


