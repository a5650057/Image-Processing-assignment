import os
from PIL import Image

def convert_jpg_to_png(input_folder, output_folder):
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)

            try:
                # 打開 JPG 圖片並轉換為 PNG
                with Image.open(input_path) as img:
                    img.save(output_path, 'PNG')
                print(f"Converted: {input_path} -> {output_path}")
            except Exception as e:
                print(f"Error converting {input_path}: {e}")

if __name__ == "__main__":
    input_folder = "./"  
    output_folder = "./"  

    convert_jpg_to_png(input_folder, output_folder)
