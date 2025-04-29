import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.gan_model import SimpleGenerator, SimpleDiscriminator

class ImageDataset(Dataset):
    def __init__(self, img_dir, img_size=(320,320)):
        self.img_paths = [
            p for p in glob.glob(os.path.join(img_dir, '*'))
            if p.lower().endswith(('.jpg','.jpeg','.png','.bmp'))
        ]
        self.transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=img)
        return augmented['image']


def train_gan(data_dir, gan_ckpt, epochs=50, batch_size=16, lr=2e-4, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ImageDataset(data_dir)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, drop_last=True
    )

    gen = SimpleGenerator().to(device)
    dis = SimpleDiscriminator().to(device)

    criterion = nn.BCELoss()
    optim_g = optim.Adam(gen.parameters(), lr=lr)
    optim_d = optim.Adam(dis.parameters(), lr=lr)

    real_label_val, fake_label_val = 1.0, 0.0

    for epoch in range(1, epochs + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)
        for real_imgs in pbar:
            real_imgs = real_imgs.to(device)

            # ---- Discriminator step ----
            dis.zero_grad()
            out_real = dis(real_imgs)                         # [B,1,H,W]
            labels_real = torch.full_like(out_real, real_label_val, device=device)
            loss_d_real = criterion(out_real, labels_real)

            fake_imgs = gen(real_imgs)
            out_fake = dis(fake_imgs.detach())                # [B,1,H,W]
            labels_fake = torch.full_like(out_fake, fake_label_val, device=device)
            loss_d_fake = criterion(out_fake, labels_fake)

            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            optim_d.step()

            # ---- Generator step ----
            gen.zero_grad()
            out_fake_for_g = dis(fake_imgs)                   # [B,1,H,W]
            labels_gen = torch.full_like(out_fake_for_g, real_label_val, device=device)
            loss_g = criterion(out_fake_for_g, labels_gen)
            loss_g.backward()
            optim_g.step()

            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g.item()

            pbar.set_postfix({
                "Loss_D": f"{loss_d.item():.4f}",
                "Loss_G": f"{loss_g.item():.4f}"
            })

        avg_d = epoch_d_loss / len(loader)
        avg_g = epoch_g_loss / len(loader)
        tqdm.write(f"Epoch {epoch} Summary — Avg Loss_D: {avg_d:.4f}, Avg Loss_G: {avg_g:.4f}")

        # save generator checkpoint after each epoch
        torch.save(gen.state_dict(), gan_ckpt)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='folder containing all training images'
    )
    parser.add_argument(
        '--gan_ckpt', type=str, default='gan_generator.pth',
        help='where to save generator weights'
    )
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    train_gan(
        data_dir=args.data_dir,
        gan_ckpt=args.gan_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Remove remaining cache and reset CUDA memory usage
    torch.cuda.empty_cache()
