# ui_demo_two.py
import sys
import os
import json
import subprocess
import math

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout,
    QHBoxLayout, QWidget, QMessageBox, QGroupBox, QGridLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image, ImageDraw
from shapely.geometry import Polygon


def polygon_to_bbox(poly_points):
    """
    將4點多邊形 => (xmin, ymin, xmax, ymax) 的水平外接框
    poly_points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    回傳 (xmin, ymin, xmax, ymax)
    """
    xs = [p[0] for p in poly_points]
    ys = [p[1] for p in poly_points]
    return (min(xs), min(ys), max(xs), max(ys))


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fracture Detection and Evaluation")
        self.folder_path = ""
        self.image_list = []
        self.current_index = 0

        # 上一次「偵測舟狀骨」得到的 BBox
        self.scaphoid_bbox = None
        # 上一次「偵測骨折」得到的 BBox
        self.fracture_bbox = None
        # 舟狀骨裁切後的圖片路徑
        self.last_scaphoid_cropped_path = None

        # ---- 評估指標 ----
        self.total_TP = 0
        self.total_FP = 0
        self.total_FN = 0
        self.total_TN = 0

        # ---- Ground Truth 資料夾 ----
        self.ground_truth_dir = ""

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

        # 偵測「舟狀骨位置」的按鈕
        self.btn_scaphoid_detect = QPushButton("Scaphoid Detection")
        self.btn_scaphoid_detect.clicked.connect(self.run_scaphoid_detection)

        # 偵測「骨折」的按鈕
        self.btn_fracture_detect = QPushButton("Fracture Detection")
        self.btn_fracture_detect.clicked.connect(self.run_fracture_detection)

        # ---- 評估指標顯示 ----
        self.metrics_group = QGroupBox("Evaluation Metrics")
        metrics_layout = QGridLayout()

        self.label_iou = QLabel("IOU: N/A")
        self.label_accuracy = QLabel("Accuracy: N/A")
        self.label_precision = QLabel("Precision: N/A")
        self.label_recall = QLabel("Recall: N/A")

        # 設定字體和樣式（可選）
        font = self.label_iou.font()
        font.setPointSize(10)
        font.setBold(True)
        self.label_iou.setFont(font)
        self.label_accuracy.setFont(font)
        self.label_precision.setFont(font)
        self.label_recall.setFont(font)

        # 添加到佈局
        metrics_layout.addWidget(self.label_iou, 0, 0)
        metrics_layout.addWidget(self.label_accuracy, 0, 1)
        metrics_layout.addWidget(self.label_precision, 1, 0)
        metrics_layout.addWidget(self.label_recall, 1, 1)

        self.metrics_group.setLayout(metrics_layout)

        # ---- 佈局 ----
        layout_top = QHBoxLayout()
        layout_top.addWidget(self.btn_load_folder)
        layout_top.addWidget(self.btn_pre)
        layout_top.addWidget(self.btn_next)

        layout_middle = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.label_current_image)
        left_layout.addWidget(self.label_image_view)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.label_detected_image)

        layout_middle.addLayout(left_layout, stretch=1)
        layout_middle.addLayout(right_layout, stretch=1)

        layout_buttons = QHBoxLayout()
        layout_buttons.addWidget(self.btn_scaphoid_detect)
        layout_buttons.addWidget(self.btn_fracture_detect)

        layout_metrics = QVBoxLayout()
        layout_metrics.addWidget(self.metrics_group)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout_top)
        main_layout.addLayout(layout_middle)
        main_layout.addLayout(layout_buttons)
        main_layout.addLayout(layout_metrics)  # 新增評估指標顯示

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

            # 更新 Ground Truth 資料夾路徑
            # 這裡使用絕對路徑來避免相對路徑的問題
            self.ground_truth_dir = os.path.abspath(os.path.join(self.folder_path, "../../fracture_detection/annotations"))

            # 重置累積指標
            self.total_TP = 0
            self.total_FP = 0
            self.total_FN = 0
            self.total_TN = 0

            # 更新 UI 評估指標顯示
            self.update_metrics_ui()

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
            pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
            self.label_image_view.setPixmap(pixmap)
        else:
            self.label_image_view.setText("Could not load image")

        # 每次切換時，重置狀態
        self.scaphoid_bbox = None
        self.last_scaphoid_cropped_path = None
        self.fracture_bbox = None
        self.label_detected_image.setText("Detection Result")

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.show_image()

    # -----------------------------------------------------------
    # 建立 output 資料夾
    # -----------------------------------------------------------
    def make_output_dir(self):
        output_dir = os.path.join(self.folder_path, "outputtwo")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    # -----------------------------------------------------------
    # (1) 舟狀骨偵測
    # -----------------------------------------------------------
    def run_scaphoid_detection(self):
        """
        執行第一階段：檢測整張影像中的「舟狀骨」位置
        1) 呼叫 python image_demo.py [image] scaphoid_config scaphoid_checkpoint --out-file out.jpg --bbox-file bbox.json
        2) 取得 bbox (x1,y1,x2,y2) 從 bbox.json
        3) 裁切該ROI => output/xxx_scaphoid_crop.jpg
        4) 在右側 label_detected_image 顯示
        """
        if not self.image_list:
            QMessageBox.warning(self, "Warning", "No images loaded.")
            return

        img_name = self.image_list[self.current_index]
        full_path = os.path.join(self.folder_path, img_name)

        scaphoid_config = "configs/rotated_faster_rcnn/my_rotated_faster_rcnn_scaphoid.py"
        scaphoid_checkpoint = "work_dirs/my_rotated_faster_rcnn_scaphoid/epoch_12.pth"
        demo_script = "demo/image_demo.py"

        # 建立 output 資料夾
        output_dir = self.make_output_dir()
        # 輸出檔名
        base_name, ext = os.path.splitext(img_name)
        out_file = f"{base_name}_scaphoid_out{ext}"
        out_full_path = os.path.join(output_dir, out_file)

        # BBox JSON 檔名
        bbox_json = os.path.join(output_dir, f"{base_name}_scaphoid_bbox.json")

        cmd = [
            "python", demo_script,
            full_path,
            scaphoid_config,
            scaphoid_checkpoint,
            "--out-file", out_full_path,
            "--score-thr", "0.3",
            "--bbox-file", bbox_json
        ]
        print("Run scaphoid detection cmd:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Scaphoid Detection 命令執行失敗:\n{e}")
            return

        # 讀取 bbox.json
        try:
            with open(bbox_json, 'r') as f:
                bbox_data = json.load(f)
                if bbox_data['bbox']:
                    self.scaphoid_bbox = tuple(bbox_data['bbox'])
                else:
                    self.scaphoid_bbox = None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"讀取 BBox JSON 檔案失敗: {e}")
            return

        if self.scaphoid_bbox is None:
            QMessageBox.warning(self, "Warning", "未偵測到舟狀骨。")
            return

        # 裁切 scaphoid 區域 => output/xxx_scaphoid_crop.jpg
        try:
            img = Image.open(full_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"讀取原圖失敗: {e}")
            return

        x1, y1, x2, y2 = self.scaphoid_bbox
        cropped = img.crop((x1, y1, x2, y2))

        crop_file = f"{base_name}_scaphoid_crop{ext}"
        crop_full_path = os.path.join(output_dir, crop_file)

        try:
            cropped.save(crop_full_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"存檔失敗: {e}")
            return

        self.last_scaphoid_cropped_path = crop_full_path

        # 右側顯示 scaphoid_crop
        pixmap = QPixmap(crop_full_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
            self.label_detected_image.setPixmap(pixmap)
        else:
            self.label_detected_image.setText("無法載入 scaphoid crop 圖檔")

        QMessageBox.information(
            self, "Scaphoid Detection",
            f"舟狀骨偵測完成，已裁切 => {crop_full_path}"
        )

    # -----------------------------------------------------------
    # (2) 骨折偵測
    # -----------------------------------------------------------
    def run_fracture_detection(self):
        """
        第二階段：檢測骨折
        1) 需要先有 scaphoid_crop (若 self.last_scaphoid_cropped_path=None =>警告)
        2) 呼叫 python image_demo.py scaphoid_crop fracture_config fracture_checkpoint --out-file fracture_out.jpg --bbox-file fracture_bbox.json
        3) 根據 score 顯示結果
        4) 計算並更新 Accuracy, Recall, Precision
        5) 計算並顯示 IOU
        6) 在最終圖片上添加 Ground Truth 的黃色框框
        """
        if self.last_scaphoid_cropped_path is None:
            QMessageBox.warning(self, "Warning", "請先偵測舟狀骨位置!")
            return

        fracture_config = "configs/rotated_faster_rcnn/my_rotated_faster_rcnn_fracture.py"
        fracture_checkpoint = "work_dirs/my_rotated_faster_rcnn_fracture/epoch_12.pth"
        demo_script = "demo/image_demo.py"

        # 輸出到 output/
        output_dir = self.make_output_dir()
        base_name, ext = os.path.splitext(os.path.basename(self.last_scaphoid_cropped_path))
        out_file = f"{base_name}_fracture_out{ext}"
        out_full_path = os.path.join(output_dir, out_file)

        # BBox JSON 檔名
        fracture_bbox_json = os.path.join(output_dir, f"{base_name}_fracture_bbox.json")

        cmd = [
            "python", demo_script,
            self.last_scaphoid_cropped_path,
            fracture_config,
            fracture_checkpoint,
            "--out-file", out_full_path,
            "--score-thr", "0.3",
            "--bbox-file", fracture_bbox_json
        ]
        print("Run fracture detection cmd:", " ".join(cmd))

        try:
            # 執行命令
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Fracture Detection 命令執行失敗:\n{e}")
            return

        # 讀取 fracture_bbox_json，包含 'bbox' 和 'score'
        try:
            with open(fracture_bbox_json, 'r') as f:
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
                    self.fracture_bbox = pred_polygon
                    self.fracture_score = score
                else:
                    self.fracture_bbox = None
                    self.fracture_score = 0.0
        except Exception as e:
            QMessageBox.critical(self, "Error", f"讀取 Fracture BBox JSON 檔案失敗: {e}")
            return

        # 載入 Ground Truth
        # 將裁切後的 base_name 轉換回原始圖像名稱
        if base_name.endswith("_scaphoid_crop"):
            original_base_name = base_name[:-len("_scaphoid_crop")]
        else:
            original_base_name = base_name  # 如果命名不符合預期，則採用原始 base_name

        gt_json = os.path.join(self.ground_truth_dir, f"{original_base_name}.json")

        if not os.path.exists(gt_json):
            QMessageBox.warning(self, "Warning", f"找不到 Ground Truth 檔案: {gt_json}")
            return

        try:
            with open(gt_json, 'r') as f:
                gt_data = json.load(f)
                has_fracture = False
                gt_bbox = None
                for entry in gt_data:
                    if entry.get('name') == "Fracture" and entry.get('bbox') is not None:
                        has_fracture = True
                        gt_bbox = entry.get('bbox')  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                        break
        except Exception as e:
            QMessageBox.critical(self, "Error", f"讀取 Ground Truth JSON 檔案失敗: {e}")
            return

        # 基於分數和 Ground Truth 判斷 TP, FP, FN, TN
        score_threshold = 0.45  # 分數閾值

        # 初始化 IOU
        iou = 0.0

        # 計算 IOU (僅用於顯示，不影響 TP, FP, FN, TN)
        if has_fracture and self.fracture_bbox is not None:
            if gt_bbox:
                iou = compute_iou(gt_bbox, self.fracture_bbox)
        elif self.fracture_bbox is not None:
            # No Ground Truth fracture, but detected fracture
            iou = 0.0

        # 判斷 TP, FP, FN, TN 基於分數和 Ground Truth
        if self.fracture_score > score_threshold:
            if has_fracture:
                self.total_TP += 1
            else:
                self.total_FP += 1
        else:
            if has_fracture:
                self.total_FN += 1
            else:
                self.total_TN += 1

        # 計算 Precision 和 Recall
        if self.total_TP + self.total_FP > 0:
            precision = self.total_TP / (self.total_TP + self.total_FP)
        else:
            precision = 0.0

        if self.total_TP + self.total_FN > 0:
            recall = self.total_TP / (self.total_TP + self.total_FN)
        else:
            recall = 0.0

        if self.total_TP + self.total_FP + self.total_FN + self.total_TN > 0:
            accuracy = (self.total_TP + self.total_FN) / (self.total_TP + self.total_FP + self.total_FN + self.total_TN)
        else:
            accuracy = 0.0

        # 更新 UI 評估指標顯示
        self.label_iou.setText(f"IOU: {iou:.2f}")
        self.label_accuracy.setText(f"Accuracy: {accuracy:.2f}")
        self.label_precision.setText(f"Precision: {precision:.2f}")
        self.label_recall.setText(f"Recall: {recall:.2f}")

        # 判斷分數是否大於 0.45 並根據結果顯示
        if self.fracture_score > score_threshold:
            if self.fracture_bbox is not None:
                # 顯示 fracture_out.jpg (有 bbox)
                pixmap = QPixmap(out_full_path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
                    self.label_detected_image.setPixmap(pixmap)
                else:
                    self.label_detected_image.setText("無法載入 fracture out 圖檔")

                # 顯示 message box with score and metrics
                QMessageBox.information(
                    self, "Fracture Detection",
                    f"骨折檢測完成 => {out_full_path}\n分數: {self.fracture_score:.2f}\nIOU: {iou:.2f}\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}"
                )
        else:
            # 顯示 scaphoid_crop.jpg (無 bbox)
            pixmap = QPixmap(self.last_scaphoid_cropped_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
                self.label_detected_image.setPixmap(pixmap)
            else:
                self.label_detected_image.setText("無法載入 scaphoid crop 圖檔")

            # 顯示 message box indicating no fracture detected
            QMessageBox.information(
                self, "Fracture Detection",
                f"未偵測到骨折\nIOU: {iou:.2f}\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}"
            )

        # 將分數和 IOU 輸出到終端
        if self.fracture_score > score_threshold:
            print(f"Fracture detection score: {self.fracture_score}, IOU: {iou:.2f}")
        else:
            print(f"Fracture detection score: {self.fracture_score}, IOU: {iou:.2f}")

        # 輸出累積指標
        print(f"Total TP: {self.total_TP}, FP: {self.total_FP}, FN: {self.total_FN}, TN: {self.total_TN}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # 在偵測結果圖片上繪製 Ground Truth 的黃色框框
        if gt_bbox:
            # 調用繪製 Ground Truth 框框的函數，使用多邊形
            combined_image_path = os.path.join(output_dir, f"{base_name}_combined{ext}")
            gt_polygon = [tuple(point) for point in gt_bbox]  # Convert to list of tuples

            self.draw_ground_truth_bbox(
                image_path=out_full_path,
                gt_polygon=gt_polygon,
                out_file=combined_image_path
            )

            # 更新右側顯示為包含 Ground Truth 框框的圖片
            pixmap = QPixmap(combined_image_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
                self.label_detected_image.setPixmap(pixmap)
            else:
                self.label_detected_image.setText("無法載入 combined 圖檔")

            # 更新訊息框以反映 Ground Truth 框框的添加
            QMessageBox.information(
                self, "Fracture Detection",
                f"Ground Truth 框框已添加至 {combined_image_path}"
            )


    # -----------------------------------------------------------
    # 繪製 Ground Truth 框框的函數
    # -----------------------------------------------------------
    def draw_ground_truth_bbox(self, image_path, gt_polygon, out_file=None):
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


    # -----------------------------------------------------------
    # 更新 UI 評估指標顯示
    # -----------------------------------------------------------
    def update_metrics_ui(self):
        """
        更新 UI 上的評估指標標籤。
        """
        if self.total_TP == 0 and self.total_FP == 0 and self.total_FN == 0 and self.total_TN == 0:
            # 初始狀態
            self.label_iou.setText("IOU: N/A")
            self.label_accuracy.setText("Accuracy: N/A")
            self.label_precision.setText("Precision: N/A")
            self.label_recall.setText("Recall: N/A")
        else:
            # 計算 Precision 和 Recall
            if self.total_TP + self.total_FP > 0:
                precision = self.total_TP / (self.total_TP + self.total_FP)
            else:
                precision = 0.0

            if self.total_TP + self.total_FN > 0:
                recall = self.total_TP / (self.total_TP + self.total_FN)
            else:
                recall = 0.0

            if self.total_TP + self.total_FP + self.total_FN + self.total_TN > 0:
                accuracy = (self.total_TP + self.total_TN) / (self.total_TP + self.total_FP + self.total_FN + self.total_TN)
            else:
                accuracy = 0.0

            # 更新標籤
            self.label_accuracy.setText(f"Accuracy: {accuracy:.2f}")
            self.label_precision.setText(f"Precision: {precision:.2f}")
            self.label_recall.setText(f"Recall: {recall:.2f}")
            self.label_iou.setText(f"IOU: N/A")  # IOU 是針對單張圖片，不累積


# -----------------------------------------------------------
# 程式入口
# -----------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
