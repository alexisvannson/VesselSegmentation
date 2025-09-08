import cv2
import os
import imageio
#from typing import List, Dict, Tuple, Optional

def perform_image_rotation(image_path: str, rotation_amount: int, output_dir: str):
    """perform rotation on image. Specific to the DRIVE dataset, handling the .tif images
    and the labels are in .gif which are handled in the except with mageio.mimread (need only one frame since the gif is static)
    """
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if(image_path.split('.')[1] == 'gif'):
        gif = imageio.mimread(image_path)  # take the first frame (numpy arrays)
        frame = gif[0]
        img = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    (h, w) = frame.shape[:2] if not img.any() else img.shape[:2]  
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotation_amount, scale = 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    base_name = os.path.basename(image_path)
    base_name = os.path.splitext(base_name)[0]  # removes extension
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_rotated_by{rotation_amount}.jpg"), rotated)
    print(os.path.join(output_dir, f"{base_name}_rotated_by{rotation_amount}.jpg"))
    
def perform_data_augmentation(image_dir, label_dir, output_dir):
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    os.makedirs(f'{output_dir}/1st_manual', exist_ok=True)
    
    encoded_image_dir = os.fsencode(image_dir)
    encoded_label_dir = os.fsencode(label_dir)
    
    for image, label in zip(os.listdir(encoded_image_dir), os.listdir(encoded_label_dir)):
        image_filename, label_filename = os.fsdecode(image), os.fsdecode(label)
        for i in range(1,360):
            perform_image_rotation(image_dir + '/' + image_filename, rotation_amount=i, output_dir=f'{output_dir}/images')
            perform_image_rotation(label_dir + '/' + label_filename , rotation_amount=i, output_dir=f'{output_dir}/1st_manual')
            

if __name__ == "__main__":
    perform_data_augmentation(image_dir='DRIVE/training/images', label_dir='DRIVE/training/1st_manual', output_dir='trainig_data')
