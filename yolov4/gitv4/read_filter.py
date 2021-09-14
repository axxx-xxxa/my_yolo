import glob
import cv2
import xml.etree.ElementTree as ET

# color =


def draw(xml_root, img_root):
    xml_list = glob.glob(f'{xml_root}/*')
    print(xml_root)
    for i, xml_name in enumerate(xml_list):
        folder_index = img_root[5]
        if i % 1 == 0 and folder_index != '5':
            tree = ET.parse(xml_name)
            img_file_path = tree.findtext('./filename')
            img = cv2.imread( img_root + '/' + img_file_path)
            # print("../" + img_root + '/' + img_file_path)


            for obj in tree.iter('object'):
                gt = obj.findtext('name')
                print(gt)
                for box in obj.iter('bndbox'):
                    xmin = int(box.findtext('xmin'))
                    ymin = int(box.findtext('ymin'))
                    xmax = int(box.findtext('xmax'))
                    ymax = int(box.findtext('ymax'))
                    coor = [(xmin,ymin),(xmax,ymax)]
                    cv2.rectangle(img, coor[0], coor[1], (255, 255, 0), 2, 2)
                    cv2.putText(img, gt, coor[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (0, 0, 255), 2)
            # print(f"data/{img_root}_{i}.jpg")
            cv2.imwrite(f"data_val/{folder_index}_{i}.jpg",img)
            # tree.write(f'label/{folder_index}_{i}',encoding='UTF-8')
        else :
            if i % 500 == 0:
                tree = ET.parse(xml_name)
                img_file_path = tree.findtext('./filename')
                img = cv2.imread(img_root + '/' + img_file_path)

                for obj in tree.iter('object'):
                    gt = obj.findtext('name')
                    for box in obj.iter('bndbox'):
                        xmin = int(box.findtext('xmin'))
                        ymin = int(box.findtext('ymin'))
                        xmax = int(box.findtext('xmax'))
                        ymax = int(box.findtext('ymax'))
                        coor = [(xmin, ymin), (xmax, ymax)]
                        cv2.rectangle(img, coor[0], coor[1], (255, 255, 0), 2, 2)
                        cv2.putText(img, gt, coor[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                # print(f"data/{img_root}_{i}.jpg")
                cv2.imwrite(f"data_val/{folder_index}_{i}.jpg", img)





    return 0


def drawpng(xml_root, img_root):
    xml_list = glob.glob(f'{xml_root}/*')
    print(xml_root)
    for i, xml_name in enumerate(xml_list):
        folder_index = img_root[5]
        if i % 1 == 0 and folder_index != '5':
            tree = ET.parse(xml_name)
            img_file_path = tree.findtext('./filename')
            img_path = img_root + '/' + (img_file_path).replace('.jpg', '.png')
            img = cv2.imread(img_path)
            print(img.shape)
            # print("../" + img_root + '/' + img_file_path)


            for obj in tree.iter('object'):
                gt = obj.findtext('name')
                print(gt)
                for box in obj.iter('bndbox'):
                    xmin = int(box.findtext('xmin'))
                    ymin = int(box.findtext('ymin'))
                    xmax = int(box.findtext('xmax'))
                    ymax = int(box.findtext('ymax'))
                    coor = [(xmin,ymin),(xmax,ymax)]
                    cv2.rectangle(img, coor[0], coor[1], (255, 255, 0), 2, 2)
                    cv2.putText(img, gt, coor[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75 , (0, 0, 255), 2)
            # print(f"data/{img_root}_{i}.jpg")
            cv2.imwrite(f"data_val/{folder_index}_{i}.jpg",img)
            # tree.write(f'label/{folder_index}_{i}',encoding='UTF-8')





    return 0







if __name__=="__main__":
    xml_roots = ['data/5/out-labelimg']
    img_roots = ['data/5/out']

    for xml_root,img_root in zip(xml_roots,img_roots):
        draw(xml_root, img_root)
