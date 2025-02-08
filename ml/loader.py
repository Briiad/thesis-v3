from torch.utils.data import DataLoader, random_split
from dataset import AcneDataset

def get_loaders(data_dir, batch_size, train_ratio=0.8, transform_train=None, transform_val=None):
    """
    Creates training and validation dataloaders.
    
    Args:
        data_dir (str): Path to the dataset folder.
        batch_size (int): Batch size.
        train_ratio (float): Ratio of images used for training.
        transform_train: Transformations for training set.
        transform_val: Transformations for validation set.
    
    Returns:
        train_loader, val_loader: Dataloaders for training and validation.
    """
    # Create the full dataset with training transform (temporarily)
    full_dataset = AcneDataset(data_dir, transform=transform_train)
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Override transform for validation if provided
    if transform_val is not None:
        val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
