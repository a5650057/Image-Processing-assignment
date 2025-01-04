import os
import glob

def rename_spaces_in_dir(folder):
    """
    掃描 folder 下所有檔案，只要檔名裡有空格，就把空格改成 '-'
    """
    if not os.path.isdir(folder):
        print(f"[提示] 資料夾不存在：{folder}，跳過")
        return

    for filepath in glob.glob(os.path.join(folder, '*')):
        base = os.path.basename(filepath)  # 原檔名
        if ' ' in base:
            new_base = base.replace(' ', '-')
            new_path = os.path.join(folder, new_base)
            if os.path.exists(filepath) and not os.path.exists(new_path):
                print(f"Rename: {base} => {new_base}")
                os.rename(filepath, new_path)

def main():
    """
    一次性處理下列資料夾：
    1) scaphoid_detection/images/
    2) scaphoid_detection/annotations/
    3) fracture_detection/images/
    4) fracture_detection/annotations/

    只要檔名中有空格，就將其改成 '-'
    """

    folders_to_rename = [
        'scaphoid_detection/images',
        'scaphoid_detection/annotations',
        'fracture_detection/images',
        'fracture_detection/annotations'
    ]
    for folder in folders_to_rename:
        rename_spaces_in_dir(folder)

    print("=== 所有空格檔名已處理完畢 ===")


if __name__ == "__main__":
    main()
