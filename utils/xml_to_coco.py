import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}  # Automatically generated from labels

def get_category_id(name):
    if name not in PRE_DEFINE_CATEGORIES:
        PRE_DEFINE_CATEGORIES[name] = len(PRE_DEFINE_CATEGORIES) + 1
    return PRE_DEFINE_CATEGORIES[name]

def convert(xml_dir, json_file):
    image_id = 0
    bbox_id = START_BOUNDING_BOX_ID
    coco = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }

    for xml_file in tqdm(os.listdir(xml_dir)):
        if not xml_file.endswith(".xml"):
            continue
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename")
        size = root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))

        image_id += 1
        coco["images"].append({
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id
        })

        for obj in root.findall("object"):
            name = obj.findtext("name")
            category_id = get_category_id(name)

            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.findtext("xmin")))
            ymin = int(float(bndbox.findtext("ymin")))
            xmax = int(float(bndbox.findtext("xmax")))
            ymax = int(float(bndbox.findtext("ymax")))
            o_width = xmax - xmin
            o_height = ymax - ymin

            coco["annotations"].append({
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bbox_id,
                "ignore": 0,
                "segmentation": []
            })
            bbox_id += 1

    for name, cid in PRE_DEFINE_CATEGORIES.items():
        coco["categories"].append({"supercategory": "none", "id": cid, "name": name})

    with open(json_file, 'w') as f:
        json.dump(coco, f, indent=4)
