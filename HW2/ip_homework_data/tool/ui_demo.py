# modified_demo.py
import sys
import os
import json
import subprocess
import math

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QTextEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image, ImageDraw
from shapely.geometry import Polygon


def rotated_box_to_polygon(rotated_box):
    """
    Convert rotated box [cx, cy, w, h, angle] to polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    if len(rotated_box) < 5:
        raise ValueError("Rotated box must have at least 5 elements: [cx, cy, w, h, angle]")
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


def compute_iou(gt_polygon, pred_polygon):
    """
    計算旋轉矩形的 IOU。
    gt_polygon: list of [x, y] points for Ground Truth
    pred_polygon: list of [x, y] points for Predicted BBox
    回傳 IOU (0~1)
    """
    try:
        gt_poly = Polygon(gt_polygon)
        pred_poly = Polygon(pred_polygon)
        
        if not gt_poly.is_valid or not pred_poly.is_valid:
            return 0.0
        
        inter_area = gt_poly.intersection(pred_poly).area
        union_area = gt_poly.union(pred_poly).area
        
        if union_area == 0:
            return 0.0
        
        iou = inter_area / union_area
        return iou
    except Exception as e:
        print(f"Error computing IOU: {e}")
        return 0.0


def draw_ground_truth_bbox(image_path, gt_polygon, out_file=None):
    """
    在圖片上繪製 Ground Truth 的黃色旋轉框框，並可選擇保存圖片。

    :param image_path: 原始圖片路徑
    :param gt_polygon: list of [x, y] points for the polygon
    :param out_file: 保存的輸出圖片路徑（可選）
    """
    if gt_polygon is None:
        print("No Ground Truth bounding box to draw.")
        return

    # 打開圖片
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # 繪製黃色旋轉框框
    draw.polygon(gt_polygon, outline="yellow", width=3)

    # 保存圖片（如果指定了 out_file）
    if out_file:
        image.save(out_file)
        print(f"Ground Truth bbox added and image saved to {out_file}")
    else:
        # 如果未指定保存路徑，僅顯示圖片
        image.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fracture Detection and Evaluation")
        self.folder_path = ""
        self.image_list = []
        self.current_index = 0

        # 用來記住「上一次 Process (裁切) 後的檔案路徑」
        self.last_cropped_path = None

        # 在此存「最後一次推論得到的 bbox」，示範為 list of [x, y] points
        # 若沒有偵測到，就設 None
        self.last_predicted_bbox = None
        self.last_predicted_score = 0.0

        # ---- UI 元件 ----
        # 1) Load Folder
        self.btn_load_folder = QPushButton("Load Folder")
        self.btn_load_folder.clicked.connect(self.load_folder)

        # 2) Pre / Next
        self.btn_pre = QPushButton("Pre")
        self.btn_next = QPushButton("Next")
        self.btn_pre.clicked.connect(self.show_previous_image)
        self.btn_next.clicked.connect(self.show_next_image)

        # 左側: 原圖
        self.label_current_image = QLabel("Current Image :")
        self.label_image_view = QLabel("No image")
        self.label_image_view.setAlignment(Qt.AlignCenter)

        # 右側: 偵測結果
        self.label_detected_image = QLabel("Detection Result")
        self.label_detected_image.setAlignment(Qt.AlignCenter)

        # 3) BBox 輸入 (四個數字)
        self.annotation_edit = QTextEdit()
        self.annotation_edit.setPlaceholderText(
            "請輸入四個數字 (x1 y1 x2 y2)\n例如:\n751 794 873 945"
        )

        # 4) Process 按鈕
        self.btn_process = QPushButton("Process")
        self.btn_process.clicked.connect(self.process_bbox)

        # 5) Detection 按鈕
        self.btn_detection = QPushButton("Detection")
        self.btn_detection.clicked.connect(self.run_detection)

        # 6) 評估分數顯示
        self.result_label = QLabel("IoU / Accuracy / Recall / Precision => ???")
        self.result_label.setAlignment(Qt.AlignLeft)

        # ---- 佈局 ----
        # (A) Top: load folder + pre/next
        layout_top = QHBoxLayout()
        layout_top.addWidget(self.btn_load_folder)
        layout_top.addWidget(self.btn_pre)
        layout_top.addWidget(self.btn_next)

        # (B) Middle: 左邊(原圖) + 右邊(偵測結果)
        layout_middle = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.label_current_image)
        left_layout.addWidget(self.label_image_view)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.label_detected_image)

        layout_middle.addLayout(left_layout, stretch=1)
        layout_middle.addLayout(right_layout, stretch=1)

        # (C) Bottom: annotation + process + detection + metrics
        layout_anno = QVBoxLayout()
        layout_anno.addWidget(self.annotation_edit)
        layout_anno.addWidget(self.btn_process)
        layout_anno.addWidget(self.btn_detection)
        layout_anno.addWidget(self.result_label)  # 用來顯示IoU/Accuracy/Recall/Precision

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout_top)
        main_layout.addLayout(layout_middle)
        main_layout.addLayout(layout_anno)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # -----------------------------------------------------------
    # 載入資料夾
    # -----------------------------------------------------------
    def load_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Folder", "./")
        if dir_path:
            self.folder_path = dir_path
            valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
            self.image_list = [
                f for f in os.listdir(dir_path)
                if f.lower().endswith(valid_ext)
            ]
            self.image_list.sort()
            self.current_index = 0
            self.show_image()
            # 清空右側顯示
            self.label_detected_image.setText("Detection Result")
            # 重置
            self.last_cropped_path = None
            self.last_predicted_bbox = None
            self.last_predicted_score = 0.0
            self.result_label.setText("IoU / Accuracy / Recall / Precision => ???")

    # -----------------------------------------------------------
    # 顯示左側當前圖片 (原圖)
    # -----------------------------------------------------------
    def show_image(self):
        if not self.image_list:
            self.label_current_image.setText("No images in folder.")
            self.label_image_view.setText("No image")
            return

        img_name = self.image_list[self.current_index]
        full_path = os.path.join(self.folder_path, img_name)
        self.label_current_image.setText(f"Current Image : {img_name}")

        pixmap = QPixmap(full_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
            self.label_image_view.setPixmap(pixmap)
        else:
            self.label_image_view.setText("Could not load image")

        # 每次切換圖片都把 last_* 清空，避免誤用舊檔
        self.last_cropped_path = None
        self.last_predicted_bbox = None
        self.last_predicted_score = 0.0
        self.result_label.setText("IoU / Accuracy / Recall / Precision => ???")

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()
            # 清空偵測結果
            self.label_detected_image.setText("Detection Result")

    def show_next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.show_image()
            # 清空偵測結果
            self.label_detected_image.setText("Detection Result")

    # -----------------------------------------------------------
    # Process: 只裁切 => output/xxx_bbox.jpg
    # -----------------------------------------------------------
    def process_bbox(self):
        if not self.image_list:
            QMessageBox.warning(self, "Warning", "No images loaded.")
            return

        img_name = self.image_list[self.current_index]
        full_path = os.path.join(self.folder_path, img_name)

        text = self.annotation_edit.toPlainText().strip()
        parts = text.split()
        if len(parts) != 4:
            QMessageBox.warning(self, "Warning", "請輸入四個數字 (x1 y1 x2 y2)")
            return

        try:
            x1, y1, x2, y2 = map(int, parts)
        except ValueError:
            QMessageBox.warning(self, "Warning", "無法將輸入轉為整數")
            return

        # 開圖
        try:
            img = Image.open(full_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"讀圖失敗: {e}")
            return

        if img.mode != "RGB":
            img = img.convert("RGB")

        cropped = img.crop((x1, y1, x2, y2))

        # 輸出到 output/xxx_bbox.jpg
        output_dir = os.path.join(self.folder_path, "output")
        os.makedirs(output_dir, exist_ok=True)

        base_name, ext = os.path.splitext(img_name)
        out_name = f"{base_name}_bbox{ext}"
        out_full_path = os.path.join(output_dir, out_name)

        try:
            cropped.save(out_full_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"存檔失敗: {e}")
            return

        self.last_cropped_path = out_full_path

        QMessageBox.information(
            self, "Process Done", f"已輸出: {out_full_path}"
        )

    # -----------------------------------------------------------
    # Detection: 針對 output/xxx_bbox.jpg 做偵測
    # 並計算分數
    # -----------------------------------------------------------
    def run_detection(self):
        if self.last_cropped_path is None:
            QMessageBox.warning(
                self, "Warning",
                "尚未做 Process!\n請先輸入 BBox 並按下Process後再偵測。"
            )
            return

        if not os.path.exists(self.last_cropped_path):
            QMessageBox.warning(self, "Warning", f"檔案不存在:\n{self.last_cropped_path}")
            return

        # -------------------(A) 執行偵測-------------------
        base_name, ext = os.path.splitext(os.path.basename(self.last_cropped_path))
        out_file = f"{base_name}_out{ext}"

        output_dir = os.path.dirname(self.last_cropped_path)  # "output"
        out_full_path = os.path.join(output_dir, out_file)

        config_file = "configs/rotated_faster_rcnn/my_rotated_faster_rcnn.py"
        checkpoint_file = "work_dirs/my_rotated_faster_rcnn_fracture/epoch_12.pth"
        demo_script = "demo/image_demo.py"

        bbox_json_path = os.path.join(output_dir, f"{base_name}_bbox.json")

        cmd = [
            "python", demo_script,
            self.last_cropped_path,
            config_file,
            checkpoint_file,
            "--out-file", out_full_path,
            "--score-thr", "0.3",
            "--bbox-file", bbox_json_path
        ]

        print("Run detection cmd:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Detection 命令執行失敗:\n{e}")
            return

        # 讀取 bbox.json
        try:
            with open(bbox_json_path, 'r') as f:
                bbox_data = json.load(f)
                if bbox_data.get('bbox') is not None:
                    bbox = bbox_data['bbox']
                    score = float(bbox_data.get('score', 0.0))
                    if len(bbox) == 4:
                        # [x1, y1, x2, y2]
                        pred_polygon = [
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[2], bbox[3]],
                            [bbox[0], bbox[3]]
                        ]
                    elif len(bbox) == 5:
                        # [cx, cy, w, h, angle]
                        pred_polygon = rotated_box_to_polygon(bbox[:5])
                    elif len(bbox) == 6:
                        # [cx, cy, w, h, angle, score]
                        pred_polygon = rotated_box_to_polygon(bbox[:5])
                    else:
                        raise ValueError(f"Unsupported bbox format with length {len(bbox)}: {bbox}")
                    self.last_predicted_bbox = pred_polygon
                    self.last_predicted_score = score
                else:
                    self.last_predicted_bbox = None
                    self.last_predicted_score = 0.0
        except Exception as e:
            QMessageBox.critical(self, "Error", f"讀取 bbox.json 檔案失敗: {e}")
            return

        # 在右側顯示檔案
        pixmap = QPixmap(out_full_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
            self.label_detected_image.setPixmap(pixmap)
        else:
            self.label_detected_image.setText("無法載入輸出圖檔")

        QMessageBox.information(
            self, "Detection Done",
            f"偵測完成，輸出檔: {out_file}\n預測 bbox={self.last_predicted_bbox}"
        )

        # -------------------(B) 計算評估指標-------------------
        # 從 ground truth 目錄讀 JSON
        # GT path: ../fracture_detection/annotations/xxxxx.json
        # 假設圖片檔名 =  "00075616-AP0.jpg" => GT json = "00075616-AP0.json"
        original_img_name = self.image_list[self.current_index]  # e.g. 00075616-AP0.jpg
        gt_name, _ = os.path.splitext(original_img_name)
        gt_json_path = os.path.join(
            "..", "fracture_detection", "annotations",
            gt_name + ".json"
        )
        if not os.path.exists(gt_json_path):
            QMessageBox.warning(self, "Warning",
                f"找不到 GT JSON:\n{gt_json_path}")
            return

        with open(gt_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # data 可能像: [{"name": "Fracture", "bbox": [[59.0,108.8],[97.8,68.6],..., [...]]}]
        # 或者 [{"name": null, "bbox": null, ...}]

        if len(data) == 0:
            QMessageBox.warning(self, "Warning", f"GT JSON內容是空的?")
            return

        # 取第一筆(示範)
        gt_item = data[0]
        gt_name = gt_item.get("name", None)
        gt_bbox_points = gt_item.get("bbox", None)  # 4點: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

        # 分析 GT 是否有骨折
        has_fracture = (gt_name is not None and gt_bbox_points is not None)

        # 分析 predict 是否有偵測
        predict_bbox = self.last_predicted_bbox  # list of [x, y] or None
        has_pred = (predict_bbox is not None)

        iou_val = 0.0
        TP = FP = FN = TN = 0

        if has_fracture:
            # 轉成多邊形
            gt_polygon = gt_bbox_points  # Ground Truth 多邊形
            if has_pred:
                iou_val = compute_iou(gt_polygon, predict_bbox)
                # 如果 iou >= 0.5 => TP，否則 => FP
                if iou_val >= 0.5:
                    TP = 1
                else:
                    FP = 1
            else:
                # Ground Truth 有骨折，但 predict 沒框 => FN
                FN = 1
        else:
            # GT 沒有骨折
            if has_pred:
                # 如果 predict 有框 => FP
                FP = 1
            else:
                # 都沒框 => TN
                TN = 1

        # 算 Accuracy, Recall, Precision
        denom = (TP + FP + FN + TN)
        accuracy = (TP + FN) / denom if denom > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        # 以百分比顯示
        iou_str = f"{iou_val:.3f}"
        acc_str = f"{(accuracy * 100):.2f}%"
        rec_str = f"{(recall * 100):.2f}%"
        pre_str = f"{(precision * 100):.2f}%"

        msg = (f"IoU={iou_str}, "
               f"Accuracy={acc_str}, "
               f"Recall={rec_str}, "
               f"Precision={pre_str}")
        self.result_label.setText(msg)

        # 繪製 Ground Truth 框框
        if has_fracture:
            combined_image_path = os.path.join(output_dir, f"{base_name}_combined{ext}")
            gt_polygon = [tuple(point) for point in gt_bbox_points]  # Convert to list of tuples

            draw_ground_truth_bbox(
                image_path=out_full_path,
                gt_polygon=gt_polygon,
                out_file=combined_image_path
            )

            # 更新右側顯示為包含 Ground Truth 框框的圖片
            pixmap = QPixmap(combined_image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
                self.label_detected_image.setPixmap(pixmap)
            else:
                self.label_detected_image.setText("無法載入 combined 圖檔")

            # 更新訊息框以反映 Ground Truth 框框的添加
            QMessageBox.information(
                self, "Fracture Detection",
                f"Ground Truth 框框已添加至 {combined_image_path}"
            )


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
