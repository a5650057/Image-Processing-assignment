# demo/image_demo.py
import json
import math
from argparse import ArgumentParser

import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import mmrotate  # noqa: F401


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--bbox-file', default=None, help='Path to save bbox coordinates in JSON')
    args = parser.parse_args()
    return args


def rotated_box_to_polygon(rotated_box):
    """
    Convert rotated box [cx, cy, w, h, angle] to polygon [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    cx, cy, w, h, angle = rotated_box[:5]
    angle_rad = math.radians(angle)

    # Calculate the corner points relative to center
    dx = w / 2
    dy = h / 2

    # Four corners before rotation
    corners = [
        (-dx, -dy),
        (dx, -dy),
        (dx, dy),
        (-dx, dy)
    ]

    rotated_corners = []
    for x, y in corners:
        xr = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        yr = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        rotated_corners.append([xr + cx, yr + cy])

    return rotated_corners


def get_axis_aligned_bbox_from_rotated_box(rotated_box):
    """
    Get axis-aligned bbox [x1, y1, x2, y2] from rotated box
    """
    polygon = rotated_box_to_polygon(rotated_box)
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return [int(x1), int(y1), int(x2), int(y2)]


def keep_top1_across_all_classes(detection_result, score_thr=0.3):
    """
    保留所有類別中分數最高的一個框
    """
    if not isinstance(detection_result, (list, tuple)):
        return detection_result  # 有些情況下可能是其他格式

    all_bboxes = []
    for cls_idx, bboxes in enumerate(detection_result):
        if not isinstance(bboxes, np.ndarray) or bboxes.size == 0:
            continue
        keep = bboxes[bboxes[:, -1] >= score_thr]
        if keep.size == 0:
            continue
        for row in keep:
            score = row[-1]
            all_bboxes.append((score, cls_idx, row))

    if len(all_bboxes) == 0:
        new_result = []
        for cidx, bboxes in enumerate(detection_result):
            shape = (0, bboxes.shape[1]) if isinstance(bboxes, np.ndarray) else (0, 5)
            new_result.append(np.zeros(shape, dtype=np.float32))
        return new_result

    # 找出分數最高者
    all_bboxes.sort(key=lambda x: x[0], reverse=True)
    best_score, best_cls, best_box = all_bboxes[0]

    # 只保留這個 best_box，其他清空
    new_result = []
    for cidx, bboxes in enumerate(detection_result):
        if cidx == best_cls:
            best_box_2d = best_box[None, :]
            new_result.append(best_box_2d)
        else:
            shape = (0, bboxes.shape[1]) if isinstance(bboxes, np.ndarray) else (0, 5)
            new_result.append(np.zeros(shape, dtype=np.float32))
    return new_result


def main(args):
    # 1) 建立模型
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # 2) 進行推論
    result = inference_detector(model, args.img)
    # 3) 只保留分數最高的框
    filtered_result = keep_top1_across_all_classes(result, args.score_thr)
    # 4) 取得偵測框坐標
    bbox = None
    score = 0.0  # 初始化分數
    for cls_bboxes in filtered_result:
        if len(cls_bboxes) > 0:
            bbox = cls_bboxes[0]
            break
    if bbox is not None:
        # 調試輸出，查看 bbox 的內容和長度
        print(f"Detected bbox: {bbox}, Length: {len(bbox)}")

        if len(bbox) == 6:
            # 假設格式為 [cx, cy, w, h, angle, score]
            axis_aligned_bbox = get_axis_aligned_bbox_from_rotated_box(bbox)
            score = float(bbox[5])  # 確保轉換為 Python float
        elif len(bbox) == 5:
            # 假設格式為 [x1, y1, x2, y2, score]
            x1, y1, x2, y2, score = bbox
            axis_aligned_bbox = [int(x1), int(y1), int(x2), int(y2)]
            score = float(score)  # 確保轉換為 Python float
        else:
            raise ValueError(f"Unsupported bbox format with length {len(bbox)}: {bbox}")
    else:
        axis_aligned_bbox = None
        score = 0.0

    # 5) 保存 bbox 坐標和分數到 JSON
    if args.bbox_file:
        with open(args.bbox_file, 'w') as f:
            if axis_aligned_bbox is not None:
                json.dump({'bbox': axis_aligned_bbox, 'score': score}, f)
                print(f"BBox coordinates and score saved to {args.bbox_file}")
            else:
                json.dump({'bbox': None, 'score': 0.0}, f)
                print(f"No bbox detected. Saved to {args.bbox_file}")

    # 6) 顯示並保存結果圖
    show_result_pyplot(
        model,
        args.img,
        filtered_result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
