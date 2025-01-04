<!-- 
conda create -n open-mmlab python=3.10 pytorch==1.10.1 cudatoolkit=10.2 torchvision -c pytorch -y


pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html

pip install yapf==0.31.0 -->




cd 到 HW2/ip_homework_data 後 git clone https://github.com/open-mmlab/mmrotate


conda建立虛擬環境:

conda env create -f environment.yml


1.

將ip_homework_data/tool/裡面的

ui_demo_two.py
ui_demo.py

都放到mmrotate資料夾裡

2.

my_rotated_faster_rcnn_fracture.py
my_rotated_faster_rcnn_scaphoid.py

放到mmrotate/configs/rotated_faster_rcnn裡

3.

image_demo.py

放到mmrotate/demo 裡

4.
舟狀骨偵測訓練:
python tools/train.py configs/rotated_faster_rcnn/my_rotated_faster_rcnn_scaphoid.py


骨折偵測訓練:
python tools/train.py configs/rotated_faster_rcnn/my_rotated_faster_rcnn_fracture.py


(這裡我是全部訓練 並沒有使用驗證集 因為我認為相同一批資料 應該不會有太多的變化性 所以就去擬和現在的資料集)

---
Start: cd 到mmrotate資料夾

python ui_demo_two.py