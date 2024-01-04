import torch
import os
import cv2
import numpy as np
from PIL import Image
import supervision as sv

CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
# print(sam)
mask_generator = SamAutomaticMaskGenerator(sam)
mask_predictor = SamPredictor(sam)
box_annotator = sv.BoxAnnotator(color=sv.Color.red())
mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)


def segment_image(pil_image: Image.Image, box:np.ndarray):
    
    open_cv_image = np.array(pil_image.convert('RGB'))
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    image_rgb = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    sam_result = mask_generator.generate(image_rgb)
    # mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image_rgb)

    masks, scores, logits = mask_predictor.predict(
        box=box
    )
    # detections = sv.Detections(
    #     xyxy=sv.mask_to_xyxy(masks=masks),
    #     mask=masks
    #     )
    
    for i, mask in enumerate(masks):
        if i==2:
            mask = np.array(mask * 255, dtype=np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(image_rgb)
            mask = Image.fromarray(mask)
            # new_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
            new_img = Image.new("RGBA", img.size, (255, 255, 255))

            # Görüntünün istediğiniz bölümlerini maskeye göre seçin ve kopyalayın
            for x in range(img.size[0]):
                for y in range(img.size[1]):
                    if mask.getpixel((x,y)) == (255, 255, 255): # Beyaz piksel
                        new_img.putpixel((x,y), img.getpixel((x,y)))
                        
            return new_img

            
    
    
