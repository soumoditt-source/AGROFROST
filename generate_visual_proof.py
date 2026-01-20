import os
import json
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_montage():
    # 1. Find the latest report and images
    reports = sorted(glob.glob('Drone image/Benkmura VF/analysis_report_*.json'), key=os.path.getmtime)
    if not reports: return
    
    with open(reports[-1], 'r') as f:
        data = json.load(f)
    
    # Locate OP3 image
    img_path = 'Drone image/Benkmura VF/Ortho/Post-SW/Post-SW.tif'
    if not os.path.exists(img_path):
        img_path = 'Drone image/Benkmura VF/Ortho Data/Post-Pitting/Post-Pitting.tif'

    print(f"Loading {img_path} for visual proof...")
    Image.MAX_IMAGE_PIXELS = None
    full_img = Image.open(img_path)
    
    # 2. Extract some Alives and Deads
    details = data['metrics']['details']
    alives = [p for p in details if p['status'] == 'alive'][:4]
    deads = [p for p in details if p['status'] == 'dead'][:4]
    
    # 3. Create Montage
    patch_size = 200
    montage = Image.new('RGB', (patch_size * 4, patch_size * 2 + 50), (30, 30, 30))
    draw = ImageDraw.Draw(montage)
    
    def add_patches(pits, row_idx, title):
        draw.text((10, row_idx * (patch_size + 25) + 5), title, fill=(255, 255, 255))
        for i, pit in enumerate(pits):
            x, y = pit['x'], pit['y']
            left, top = x - patch_size//2, y - patch_size//2
            right, bottom = x + patch_size//2, y + patch_size//2
            
            try:
                crop = full_img.crop((left, top, right, bottom)).resize((patch_size, patch_size))
                montage.paste(crop, (i * patch_size, row_idx * (patch_size + 25) + 25))
                # Status border
                color = (0, 255, 100) if pit['status'] == 'alive' else (255, 50, 50)
                cp_draw = ImageDraw.Draw(montage)
                cp_draw.rectangle([i * patch_size, row_idx * (patch_size + 25) + 25, (i+1)*patch_size, (row_idx+1)*(patch_size) + 25], outline=color, width=3)
            except:
                pass

    add_patches(alives, 0, "DETECTED ALIVE (BIO-VITALITY POSITIVE)")
    add_patches(deads, 1, "DETECTED DEAD (BIO-VITALITY NEGATIVE / EMPTY PIT)")
    
    montage.save('Drone image/visual_proof_montage.png')
    print("Visual proof saved to Drone image/visual_proof_montage.png")

if __name__ == "__main__":
    generate_montage()
