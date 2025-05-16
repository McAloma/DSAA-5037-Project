import sys, os, torch
sys.path.append("your/path")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from torchvision.models import ResNet50_Weights
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTModel
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel

from concurrent.futures import ThreadPoolExecutor


class ResNet_Encoder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.model = self.model.to(self.device)
        self.model.eval()

        # 定义图像预处理过程
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def encode(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        with torch.no_grad():
            features = self.model(image_tensor)  # shape: (1, 2048, 1, 1)
            embedding = features.view(features.size(0), -1).cpu().numpy()  # shape: (1, 2048)
        return embedding
    
    def encode_batch(self, images):
        batch_size = 50
        embeddings_list = []

        def process_batch(batch_images):
            image_tensors = torch.stack([self.transform(img) for img in batch_images], dim=0).to(self.device)
            with torch.no_grad():
                features = self.model(image_tensors)
                embeddings = features.view(features.size(0), -1).cpu().numpy()
            return embeddings

        with ThreadPoolExecutor() as executor:
            batch_images = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
            embeddings_list = list(executor.map(process_batch, batch_images))

        all_embeddings = np.vstack(embeddings_list).tolist()
        
        return all_embeddings



class ViT_Encoder:
    def __init__(self):
        model_name="google/vit-base-patch16-224-in21k"
        cache_dir = "./ckpts/vit"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()

    def encode(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
        return image_embedding

    def encode_batch(self, images):
        batch_size = 50
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            inputs = self.feature_extractor(images=batch_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0].cpu().numpy().tolist()
                all_embeddings.extend(embeddings)

        return all_embeddings


class DINO_Encoder:
    def __init__(self):
        model_name = "facebook/dinov2-base"
        cache_dir = "./ckpts/dino"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()

    def encode(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
        return image_embedding
    
    def encode_batch(self, images):
        batch_size = 1000
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0].cpu().numpy().tolist()
                all_embeddings.extend(embeddings)

        return all_embeddings




if __name__ == "__main__":
    from PIL import Image
    img_path = f"data/TCGA_thumbnail/brain/GBM/0ab3870e-3dc0-468d-8d22-409cfa8b487d.png"
    img = Image.open(img_path).convert("RGB")
    imgs = [img] * 500

    resnet_encoder = ResNet_Encoder()
    image_embedding = resnet_encoder.encode(img)
    print("Size of Resnet Encoding: ", image_embedding.shape)  # Should be (1, 2048) for ResNet-50
    image_embedding = resnet_encoder.encode_batch(imgs)
    print("Size of Resnet Encoding images: ", image_embedding.shape)  # Should be (500, 2048) for ResNet-50

    vit_encoder = ViT_Encoder()
    image_embedding = vit_encoder.encode(img)
    print("Size of ViT Encoding: ", image_embedding.shape)  # Should be (1, 768) for ViT-base
    image_embedding = vit_encoder.encode_batch(imgs)
    print("Size of ViT Encoding images: ", image_embedding.shape)  # Should be (500, 768) for ViT-base

    dino_encoder = DINO_Encoder()
    image_embedding = dino_encoder.encode(img)
    print("Size of DINO Encoding: ", image_embedding.shape)  # Should be (1, 768) for DINO
    image_embedding = dino_encoder.encode_batch(imgs)
    print("Size of DINO Encoding images: ", image_embedding.shape)  # Should be (500, 768) for DINO


    # ———————————— time test ————————————

    # from time import time
    # resnet_encoder = ResNet_Encoder()

    # begin = time()
    # image_embedding = resnet_encoder.encode(img)
    # print("Size of Resnet Encoding: ", image_embedding.shape)  # Should be (1, 2048) for ResNet-50
    # end = time()
    # print("Time taken for single image: ", end - begin)

    # begin = time()
    # image_embedding = resnet_encoder.encode_batch(imgs)
    # print("Size of Resnet Encoding: ", image_embedding.shape)  # Should be (10, 2048) for ResNet-50
    # end = time()
    # print("Time taken for batch of images: ", end - begin)
