import json
import csv


csv_file_path = r'C:/Users/Administrator/Desktop/detr-signals/signal_data2/val2017/labeled_box.csv'
output_json_path = r'instances_val2017.json'


DEFAULT_W, DEFAULT_H = 35710, 256

coco = {"images": [], "annotations": [], "categories": []}


coco["categories"].extend([
    {"id": 0, "name": "lfm",  "supercategory": "signal"},
    {"id": 1, "name": "nlfm", "supercategory": "signal"},
    {"id": 2, "name": "bpsk", "supercategory": "signal"},
    {"id": 3, "name": "fsk",  "supercategory": "signal"},
])

image_id_mapping = {}
image_meta = {}
ann_id = 1

with open(csv_file_path, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        file_name  = row['image_id']
        cat_id     = int(row['Category_id'])
        bbox       = [float(row['bbox1']), float(row['bbox2']),
                      float(row['bbox3']), float(row['bbox4'])]
        video_id   = row.get('video_id', None)
        inst_id    = row.get('inst_id', None)

        if file_name not in image_id_mapping:
            img_id = len(image_id_mapping) + 1
            image_id_mapping[file_name] = img_id
            image_meta[img_id] = {
                "file_name": file_name,
                "width": DEFAULT_W,
                "height": DEFAULT_H,
                "video_id": video_id,
                "inst_ids": set(),
            }
        else:
            img_id = image_id_mapping[file_name]

        if inst_id not in (None, ""):
            image_meta[img_id]["inst_ids"].add(inst_id)


        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": cat_id,
            "bbox": bbox,
            "area": float(bbox[2] * bbox[3]),
            "iscrowd": 0,
            "inst_id": inst_id,
            "video_id": video_id,
        })
        ann_id += 1


for img_id in sorted(image_meta.keys()):
    meta = image_meta[img_id]
    coco["images"].append({
        "id": img_id,
        "file_name": meta["file_name"],
        "width": meta["width"],
        "height": meta["height"],
        "video_id": meta["video_id"],
        "inst_ids": sorted(list(meta["inst_ids"])),
    })

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)

print(f"COCO JSON written to: {output_json_path}")
print(f"#images={len(coco['images'])}, #annotations={len(coco['annotations'])}")
