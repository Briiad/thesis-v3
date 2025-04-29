from dataset import CustomVOCDataset
from config import data_cfg

def count_annotations(dataset):
    # Initialize count dictionary for each class in your categories list
    counts = {cat: 0 for cat in dataset.categories}
    
    # Loop over each valid sample (each image-annotation pair)
    for sample in dataset.valid_samples:
        bboxes, labels = dataset._parse_annotation(sample['annotation'])
        # Each label is a number; we map it back to its category
        for label in labels:
            for cat, idx in dataset.cat_to_idx.items():
                if idx == label:
                    counts[cat] += 1
    return counts

# Create dataset instances for train, valid, and test splits
train_dataset = CustomVOCDataset(
    data_dir=data_cfg.train_dir,
    img_size=data_cfg.img_size,
    mean=data_cfg.mean,
    std=data_cfg.std,
    categories=data_cfg.categories,
    use_gan_aug=True,
    gan_ckpt=data_cfg.gan_ckpt
)

val_dataset = CustomVOCDataset(
    data_dir=data_cfg.valid_dir,
    img_size=data_cfg.img_size,
    mean=data_cfg.mean,
    std=data_cfg.std,
    categories=data_cfg.categories,
    use_gan_aug=False
)

test_dataset = CustomVOCDataset(
    data_dir=data_cfg.test_dir,
    img_size=data_cfg.img_size,
    mean=data_cfg.mean,
    std=data_cfg.std,
    categories=data_cfg.categories,
    use_gan_aug=False
)

if __name__ == '__main__':
  # Print out the counts for each split
  print("Train annotations:", count_annotations(train_dataset))
  print("Validation annotations:", count_annotations(val_dataset))
  print("Test annotations:", count_annotations(test_dataset))
