import os
import json
from PIL import Image

def extract_bbox_images(images_dir, annotations_dir, output_dir):
    """
    Extract bounding box images based on annotation JSON files.

    Args:
        images_dir (str): Path to the images folder.
        annotations_dir (str): Path to the annotations folder.
        output_dir (str): Path to save cropped images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for annotation_file in os.listdir(annotations_dir):
        if not annotation_file.endswith('.json'):
            continue

        annotation_path = os.path.join(annotations_dir, annotation_file)

        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        image_name = os.path.splitext(annotation_file)[0] + '.jpg'
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Image {image_name} not found, skipping.")
            continue

        image = Image.open(image_path)

        for idx, ann in enumerate(annotations):
            if 'bbox' not in ann or 'name' not in ann:
                print(f"Invalid annotation format in {annotation_file}, skipping annotation.")
                continue

            bbox = list(map(int, ann['bbox']))  
            cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

            output_image_name_base = os.path.splitext(image_name)[0]

            

            # Save as JPG
            output_image_path_jpg = os.path.join(output_dir, output_image_name_base + ".jpg")
            cropped_image.convert("RGB").save(output_image_path_jpg)

            print(f"Saved cropped images:{output_image_path_jpg}")

if __name__ == "__main__":
    images_directory = "scaphoid_detection/images"
    annotations_directory = "scaphoid_detection/annotations"
    output_directory = "scaphoid_detection/output"

    extract_bbox_images(images_directory, annotations_directory, output_directory)
