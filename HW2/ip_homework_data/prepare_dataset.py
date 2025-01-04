import os
import json
import argparse

def convert_scaphoid(json_dir, out_dir):
    """
    讀取 json_dir 下每個 .json (Scaphoid標註)
    格式: [ { "name": "Scaphoid", "bbox": [x1, y1, x2, y2] } ]
    x1,y1,x2,y2 可能是 int/float/string
    => 輸出 DOTA label:
       x1 y1 x2 y1 x2 y2 x1 y2 Scaphoid 0
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(json_dir):
        if not fname.lower().endswith('.json'):
            continue
        json_path = os.path.join(json_dir, fname)
        base_name, _ = os.path.splitext(fname)
        out_txt = os.path.join(out_dir, base_name + '.txt')

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(out_txt, 'w', encoding='utf-8') as fout:
            # data 可能是 list
            for item in data:
                cls_name = item.get("name", None)
                bbox = item.get("bbox", None)
                if not cls_name or not bbox:
                    # name=null 或 bbox=null => 不輸出
                    continue
                if len(bbox) != 4:
                    continue

                # 轉成 float
                try:
                    x1 = float(bbox[0])
                    y1 = float(bbox[1])
                    x2 = float(bbox[2])
                    y2 = float(bbox[3])
                except ValueError:
                    continue

                # 做水平框 => (x1,y1),(x2,y1),(x2,y2),(x1,y2)
                line = f"{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2} {cls_name} 0\n"
                fout.write(line)
        print(f"[Scaphoid OK] {fname} => {out_txt}")

def convert_fracture(json_dir, out_dir):
    """
    讀取 json_dir 下每個 .json (Fracture標註)
    格式: [ { "name": "Fracture", "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] } ]
    => 輸出 DOTA label: x1 y1 x2 y2 x3 y3 x4 y4 Fracture 0
    如果 name=null 或 bbox=null => 不輸出
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(json_dir):
        if not fname.lower().endswith('.json'):
            continue
        json_path = os.path.join(json_dir, fname)
        base_name, _ = os.path.splitext(fname)
        out_txt = os.path.join(out_dir, base_name + '.txt')

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(out_txt, 'w', encoding='utf-8') as fout:
            for item in data:
                cls_name = item.get("name", None)
                bbox = item.get("bbox", None)
                if not cls_name or not bbox:
                    
                    continue
                if len(bbox) != 4:
                    continue

                # bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                # 可能是 str or float => 轉 float
                try:
                    points = []
                    for pt in bbox:
                        x = float(pt[0])
                        y = float(pt[1])
                        points.append((x, y))
                    # points 形如 [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                except ValueError:
                    continue

                line = (f"{points[0][0]} {points[0][1]} "
                        f"{points[1][0]} {points[1][1]} "
                        f"{points[2][0]} {points[2][1]} "
                        f"{points[3][0]} {points[3][1]} "
                        f"{cls_name} 0\n")
                fout.write(line)
        print(f"[Fracture OK] {fname} => {out_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['scaphoid','fracture'],
                        help="Convert scaphoid or fracture style JSON to DOTA label.")
    parser.add_argument('--json-dir', type=str, required=True,
                        help="Directory of JSON annotations")
    parser.add_argument('--out-dir', type=str, required=True,
                        help="Output directory for labelTxt")
    args = parser.parse_args()

    if args.mode == 'scaphoid':
        convert_scaphoid(args.json_dir, args.out_dir)
    elif args.mode == 'fracture':
        convert_fracture(args.json_dir, args.out_dir)


