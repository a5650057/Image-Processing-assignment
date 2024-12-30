import os
import json
import glob
import random
import shutil
from PIL import Image

def main():
    """
    將 scaphoid_detection 與 fracture_detection 的標註整合成 COCO 格式，
    並依 80%:20% 分 train / val，最後把圖複製到 dataset/images/train 和 dataset/images/val，
    並在 dataset/annotations/ 產出 train_coco.json 與 val_coco.json。
    """

    # ----------- 0. 設定輸出路徑 -----------
    output_dir = "dataset"
    images_dir = os.path.join(output_dir, "images")
    annotations_dir = os.path.join(output_dir, "annotations")
    train_img_dir = os.path.join(images_dir, "train")
    val_img_dir = os.path.join(images_dir, "val")

    # 建立資料夾 (若不存在)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # 兩個 COCO 容器 (train, val)
    train_coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Scaphoid"},
            {"id": 2, "name": "Fracture"}
        ]
    }
    val_coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Scaphoid"},
            {"id": 2, "name": "Fracture"}
        ]
    }

    # 暫存所有影像的資訊
    # 結構: {
    #   image_name(str): {
    #       "file_name": image_name,      # 原始檔名
    #       "width": int,
    #       "height": int,
    #       "scaphoid_objs": [ (xmin, ymin, w, h), ...],
    #       "fracture_objs": [ (xmin, ymin, w, h, [x1,y1,x2,y2,x3,y3,x4,y4]), ...]
    #   }
    # }
    all_images_info = {}

    # =========== 1. 收集 Scaphoid 標註 (軸對齊) ===========
    scaphoid_json_paths = glob.glob('scaphoid_detection/annotations/*.json')
    for ann_path in scaphoid_json_paths:
        base_name = os.path.basename(ann_path)  # e.g. 00075616-AP0.json
        image_name = base_name.replace('.json', '.jpg')
        img_path = os.path.join('scaphoid_detection', 'images', image_name)

        if not os.path.exists(img_path):
            print(f"[警告] 找不到圖片: {img_path}, 跳過")
            continue

        # 讀取影像大小
        with Image.open(img_path) as img:
            w, h = img.size

        if image_name not in all_images_info:
            all_images_info[image_name] = {
                "file_name": image_name,
                "width": w,
                "height": h,
                "scaphoid_objs": [],
                "fracture_objs": []
            }

        # 讀取 JSON
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)  # e.g. [{"name": "Scaphoid", "bbox": ["751","794","873","945"]}, ...]

        for obj in ann_data:
            if obj.get("name") == "Scaphoid":
                x1, y1, x2, y2 = map(float, obj["bbox"])  # 4 個字串轉 float
                bw = x2 - x1
                bh = y2 - y1
                all_images_info[image_name]["scaphoid_objs"].append((x1, y1, bw, bh))

    # =========== 2. 收集 Fracture 標註 (四頂點) ===========
    fracture_json_paths = glob.glob('fracture_detection/annotations/*.json')
    for ann_path in fracture_json_paths:
        base_name = os.path.basename(ann_path)
        image_name = base_name.replace('.json', '.jpg')
        # 假設骨折對應的圖也在 scaphoid_detection/image/
        img_path = os.path.join('scaphoid_detection', 'images', image_name)

        if not os.path.exists(img_path):
            print(f"[警告] 找不到圖片: {img_path}, 跳過")
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        if image_name not in all_images_info:
            all_images_info[image_name] = {
                "file_name": image_name,
                "width": w,
                "height": h,
                "scaphoid_objs": [],
                "fracture_objs": []
            }

        with open(ann_path, 'r') as f:
            ann_data = json.load(f)  # [{"name":"Fracture","bbox":[[x1,y1],[x2,y2]...]}]

        for obj in ann_data:
            if obj.get("name") == "Fracture":
                pts = obj["bbox"]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                bw = xmax - xmin
                bh = ymax - ymin
                # flatten
                seg_pts = [float(coord) for point in pts for coord in point]  
                all_images_info[image_name]["fracture_objs"].append(
                    (xmin, ymin, bw, bh, seg_pts)
                )

    # =========== 3. 取得所有影像清單，打亂後分為 train / val (80:20) ===========
    all_image_names = list(all_images_info.keys())
    random.shuffle(all_image_names)
    total_count = len(all_image_names)
    train_count = int(total_count * 0.8)

    train_list = all_image_names[:train_count]
    val_list = all_image_names[train_count:]

    print(f"總共 {total_count} 張影像, 分配: Train={len(train_list)}, Val={len(val_list)}")

    # 為了 COCO 結構，維護 id 計數
    train_image_id = 1
    val_image_id = 1
    train_anno_id = 1
    val_anno_id = 1

    # =========== 4. 建立 Train COCO & 複製圖片到 dataset/images/train/ ===========
    for image_name in train_list:
        info = all_images_info[image_name]
        # 建立 image record
        train_coco["images"].append({
            "id": train_image_id,
            # 注意：後續訓練時，會在 dataset/images/train/<file_name> 讀取
            # 為了方便，我們就把 file_name 寫成單純檔名，或若需要完整相對路徑可加 'train/' 前綴
            "file_name": image_name,
            "width": info["width"],
            "height": info["height"]
        })

        # Scaphoid
        for (x1, y1, bw, bh) in info["scaphoid_objs"]:
            train_coco["annotations"].append({
                "id": train_anno_id,
                "image_id": train_image_id,
                "category_id": 1,  # Scaphoid
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            train_anno_id += 1

        # Fracture
        for (x1, y1, bw, bh, seg_pts) in info["fracture_objs"]:
            train_coco["annotations"].append({
                "id": train_anno_id,
                "image_id": train_image_id,
                "category_id": 2,  # Fracture
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "segmentation": [seg_pts]  # 多邊形
            })
            train_anno_id += 1

        # 複製檔案到 dataset/images/train/<image_name>
        src_path = os.path.join('scaphoid_detection', 'images', image_name)
        dst_path = os.path.join(train_img_dir, image_name)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

        train_image_id += 1

    # =========== 5. 建立 Val COCO & 複製圖片到 dataset/images/val/ ===========
    for image_name in val_list:
        info = all_images_info[image_name]

        val_coco["images"].append({
            "id": val_image_id,
            "file_name": image_name,
            "width": info["width"],
            "height": info["height"]
        })

        # Scaphoid
        for (x1, y1, bw, bh) in info["scaphoid_objs"]:
            val_coco["annotations"].append({
                "id": val_anno_id,
                "image_id": val_image_id,
                "category_id": 1,
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            val_anno_id += 1

        # Fracture
        for (x1, y1, bw, bh, seg_pts) in info["fracture_objs"]:
            val_coco["annotations"].append({
                "id": val_anno_id,
                "image_id": val_image_id,
                "category_id": 2,
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "segmentation": [seg_pts]
            })
            val_anno_id += 1

        # 複製檔案到 dataset/images/val/<image_name>
        src_path = os.path.join('scaphoid_detection', 'images', image_name)
        dst_path = os.path.join(val_img_dir, image_name)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

        val_image_id += 1

    # =========== 6. 寫出 train_coco.json, val_coco.json ===========
    train_coco_path = os.path.join(annotations_dir, "train_coco.json")
    val_coco_path = os.path.join(annotations_dir, "val_coco.json")

    with open(train_coco_path, 'w', encoding='utf-8') as f:
        json.dump(train_coco, f, indent=2)

    with open(val_coco_path, 'w', encoding='utf-8') as f:
        json.dump(val_coco, f, indent=2)

    print("----------------------------------------------------")
    print("完成！請查看下列路徑：")
    print(f"Train 影像: {train_img_dir}")
    print(f"Val   影像: {val_img_dir}")
    print(f"Train 標註: {train_coco_path}")
    print(f"Val   標註: {val_coco_path}")
    print("----------------------------------------------------")


if __name__ == "__main__":
    main()
