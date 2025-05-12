import os, sys, torch, timm
sys.path.append("your/path")
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor


class WSI_Image_UNI_Encoder():
    def __init__(self, param_local_dir=None):
        self.embed_model =  timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        # local_dir = "ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"    # 相对路径
        local_dir = "/hpc2ssd/JH_DATA/spooler/rsu704"                       # SSD 路径

        if param_local_dir:
            local_dir = param_local_dir
        self._device = self.infer_torch_device()
        self.embed_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True), strict=True)
        self.embed_model = self.embed_model.to(self._device)
        self.embed_model.eval()
        print("Loaded UNI Encoder.")

    def infer_torch_device(self):
        """Infer the input to torch.device."""
        try:
            has_cuda = torch.cuda.is_available()
        except NameError:
            import torch  # pants: no-infer-dep
            has_cuda = torch.cuda.is_available()
        if has_cuda:
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode_image(self, patch_image):
        patch_image = self.transform(patch_image).unsqueeze(dim=0).to(self._device)
        embedding = self.embed_model(patch_image)

        return embedding.cpu().squeeze().tolist()


    # def encode_batch(self, images, batch_size=50):
    #     def transform_one(image):
    #         return self.transform(image)

    #     all_embeddings = []

    #     for i in range(0, len(images), batch_size):
    #         batch_images = images[i:i + batch_size]

    #         with ThreadPoolExecutor() as executor:
    #             batch_tensors = list(executor.map(transform_one, batch_images))

    #         batch_tensor = torch.stack(batch_tensors, dim=0).to(self._device)

    #         with torch.no_grad():
    #             embeddings = self.embed_model(batch_tensor).cpu().squeeze().tolist()

    #         all_embeddings.extend(embeddings)

    #     return all_embeddings


    def encode_batch(self, images, batch_size=50, num_workers=4):
        class ImageDataset(Dataset):
            def __init__(self, images, transform):
                self.images = images
                self.transform = transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return self.transform(self.images[idx])

        dataset = ImageDataset(images, self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        all_embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self._device)
                embeddings = self.embed_model(batch).cpu().squeeze()

                # 保持输出一致性（列表格式）
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)

                all_embeddings.extend(embeddings.cpu().tolist())

        return all_embeddings



    
if __name__ == "__main__":
    encoder = WSI_Image_UNI_Encoder()