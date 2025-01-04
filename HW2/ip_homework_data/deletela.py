import os


folder_path = "./"  

try:
   
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"資料夾不存在: {folder_path}")

   
    for filename in os.listdir(folder_path):
        # 判斷檔名是否包含 "LA"
        if "LA" in filename:
            file_path = os.path.join(folder_path, filename)
            
            
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"已刪除檔案: {file_path}")

except Exception as e:
    print(f"發生錯誤: {e}")
