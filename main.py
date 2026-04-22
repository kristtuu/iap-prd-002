from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent
output_dir = base_dir / 'results'
output_dir.mkdir(exist_ok=True)

images = {
    'bright_image': np.array(Image.open(base_dir / 'bright_image.png').convert('RGB')),
    'dark_image': np.array(Image.open(base_dir / 'dark_image.png').convert('RGB')),
    'low_contrast_image': np.array(Image.open(base_dir / 'low_contrast_image.png').convert('RGB')),
}

def gamma_correction(img, gamma, image_name):
    augstums, platums, kanali = img.shape
    result = np.zeros((augstums, platums, kanali), dtype=np.uint8)
    
    for i in range(augstums):
        for j in range(platums):
            for k in range(kanali):
                a = img[i, j, k]
                c = 255 * (a / 255) ** gamma

                result[i, j, k] = c

    gamma_str = str(gamma).replace('.', '_')
    output_path = output_dir / f'{image_name}_gamma_{gamma_str}.png'
    Image.fromarray(result).save(output_path, 'PNG')

def linear_hist_transform(img, image_name):
    augstums, platums, kanali = img.shape
    result = np.zeros((augstums, platums, kanali), dtype=np.uint8)
    
    for k in range(kanali):
        kanal_min = np.min(img[:, :, k])
        kanal_max = np.max(img[:, :, k])

        for i in range(augstums):
            for j in range(platums):
                a = img[i, j, k]

                if kanal_max == kanal_min:
                    c = a
                else:
                    c = ((a - kanal_min) / (kanal_max - kanal_min)) * 255

                result[i, j, k] = round(c)

    output_path = output_dir / f'{image_name}_linear_hist.png'
    Image.fromarray(result).save(output_path, 'PNG')

def main():
    gamma_values = {
        'bright_image': 1.8,
        'dark_image': 0.5,
        'low_contrast_image': 0.8,
    }

    for image_name, img in images.items():
        gamma_correction(img, gamma_values[image_name], image_name)
        linear_hist_transform(img, image_name)

if __name__ == '__main__':
    main()