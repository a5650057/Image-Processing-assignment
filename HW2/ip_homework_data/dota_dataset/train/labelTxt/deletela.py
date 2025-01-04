import os

# 指定要掃描的資料夾路徑
folder_path = "./"  # 請將此處替換為你的資料夾路徑

try:
    # 確認資料夾是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"資料夾不存在: {folder_path}")

    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(folder_path):
        # 判斷檔名是否包含 "LA"
        if "LA" in filename:
            file_path = os.path.join(folder_path, filename)
            
            # 確保是檔案才刪除
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"已刪除檔案: {file_path}")

except Exception as e:
    print(f"發生錯誤: {e}")