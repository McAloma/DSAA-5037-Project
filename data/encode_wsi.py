import os, sys, json
sys.path.append("your/path")
from PIL import Image
from typing import Tuple
import numpy as np
from tqdm import tqdm

from src.tools.basic.pretrain_encoders import ResNet_Encoder, ViT_Encoder, DINO_Encoder
from src.tools.basic.wsi_encoder import WSI_Image_UNI_Encoder



def process_wsi_to_json(
        encoder, 
        wsi_image: Image.Image,
        site: str,
        subtype: str,
        patch_size: Tuple[int, int],
        step: Tuple[int, int],
        output_path: str
    ):
    width, height = wsi_image.size
    patch_w, patch_h = patch_size
    step_x, step_y = step

    patches = []
    locations = []

    for top in range(0, height - patch_h + 1, step_y):
        for left in range(0, width - patch_w + 1, step_x):
            box = (left, top, left + patch_w, top + patch_h)
            patch = wsi_image.crop(box)
            patches.append(patch)
            locations.append((left, top))

    embeddings = encoder.encode_batch(patches) 

    results = []
    for i, embedding in enumerate(embeddings):
        result = {
            "site": site,
            "subtype": subtype,
            "location": {
                "x": locations[i][0],
                "y": locations[i][1]
            },
            "embedding": embedding
        }
        results.append(result)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    data_path = f"data/TCGA_thumbnail"

    # ------ Enumerate Test ------
    sites = os.listdir(data_path)
    for site in sites:
        site_path = os.path.join(data_path, site)
        subtypes = os.listdir(site_path)
        for subtype in subtypes:
            subtype_path = os.path.join(site_path, subtype)
            images = os.listdir(subtype_path)
            for image in images:
                image_path = os.path.join(subtype_path, image)
                print(image_path)

    encoder = ResNet_Encoder()
    os.makedirs("a5037_course_work/data/embeddings/resnet", exist_ok=True)
    sites = os.listdir(data_path)
    for site in sites:
        site_path = os.path.join(data_path, site)
        subtypes = os.listdir(site_path)
        for subtype in subtypes:
            subtype_path = os.path.join(site_path, subtype)
            images = os.listdir(subtype_path)
            for image in tqdm(images, desc=f"Processing {site}/{subtype} with ResNet", ascii=True):
                image_path = os.path.join(subtype_path, image)
                try:
                    wsi_image = Image.open(image_path).convert('RGB')
                    process_wsi_to_json(
                        encoder=encoder,
                        wsi_image=wsi_image,
                        site=site,
                        subtype=subtype,
                        patch_size=(224, 224),
                        step=(224, 224),
                        output_path=f"a5037_course_work/data/embeddings/resnet/{image}.json"
                    )
                except:
                    continue
                
        #         break
        #     break
        # break

    encoder = ViT_Encoder()
    os.makedirs("a5037_course_work/data/embeddings/vit", exist_ok=True)
    sites = os.listdir(data_path)
    for site in sites:
        site_path = os.path.join(data_path, site)
        subtypes = os.listdir(site_path)
        for subtype in subtypes:
            subtype_path = os.path.join(site_path, subtype)
            images = os.listdir(subtype_path)
            for image in tqdm(images, desc=f"Processing {site}/{subtype} with ViT", ascii=True):
                image_path = os.path.join(subtype_path, image)

                try:
                    wsi_image = Image.open(image_path).convert('RGB')
                    process_wsi_to_json(
                        encoder=encoder,
                        wsi_image=wsi_image,
                        site=site,
                        subtype=subtype,
                        patch_size=(224, 224),
                        step=(224, 224),
                        output_path=f"a5037_course_work/data/embeddings/vit/{image}.json"
                    )
                except:
                    continue

        #         break
        #     break
        # break

    encoder = DINO_Encoder()
    os.makedirs("a5037_course_work/data/embeddings/dino", exist_ok=True)
    sites = os.listdir(data_path)
    for site in sites:
        site_path = os.path.join(data_path, site)
        subtypes = os.listdir(site_path)
        for subtype in subtypes:
            subtype_path = os.path.join(site_path, subtype)
            images = os.listdir(subtype_path)
            for image in tqdm(images, desc=f"Processing {site}/{subtype} with DINOv2", ascii=True):
                image_path = os.path.join(subtype_path, image)

                try:
                    wsi_image = Image.open(image_path).convert('RGB')
                    process_wsi_to_json(
                        encoder=encoder,
                        wsi_image=wsi_image,
                        site=site,
                        subtype=subtype,
                        patch_size=(224, 224),
                        step=(224, 224),
                        output_path=f"a5037_course_work/data/embeddings/dino/{image}.json"
                    )
                except:
                    continue
        #         break
        #     break
        # break


    encoder = WSI_Image_UNI_Encoder()
    os.makedirs("a5037_course_work/data/embeddings/uni", exist_ok=True)
    sites = os.listdir(data_path)
    for site in sites:
        site_path = os.path.join(data_path, site)
        subtypes = os.listdir(site_path)
        for subtype in subtypes:
            subtype_path = os.path.join(site_path, subtype)
            images = os.listdir(subtype_path)
            for image in tqdm(images, desc=f"Processing {site}/{subtype} with UNI", ascii=True):
                image_path = os.path.join(subtype_path, image)

                try:
                    wsi_image = Image.open(image_path).convert('RGB')
                    process_wsi_to_json(
                        encoder=encoder,
                        wsi_image=wsi_image,
                        site=site,
                        subtype=subtype,
                        patch_size=(224, 224),
                        step=(224, 224),
                        output_path=f"a5037_course_work/data/embeddings/uni/{image}.json"
                    )
                except:
                    continue

        #         break
        #     break
        # break