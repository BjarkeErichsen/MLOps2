"""LFW dataloading."""
import argparse
import time
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

class LFWDataset(Dataset):
    """Initialize LFW dataset."""

    def __init__(self, path_to_folder: str, transform) -> None:
        
        self.transform = transform
        self.image_paths = []
        n_folders = 0
        for root, _, files in os.walk(path_to_folder):
            n_folders += 1
            for file in files:
                #print(os.path.join(root, file))
                self.image_paths.append(os.path.join(root, file))

        print("Number of folders", n_folders)  
    def __len__(self):
        """Return length of dataset."""
        return len(self.image_paths)  # TODO: fill out

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get item from dataset."""
        img_path = self.image_paths[index]
        img = Image.open(img_path)

        # Apply transformation
        if self.transform:
            img = self.transform(img)
        return img

def make_collage(batch, width=16):
    """
    Create a collage from a batch of images in tensor format.

    :param batch: Batch of images in tensor format.
    :param width: Number of images per row in the collage.
    :return: PIL Image object representing the collage.
    """
    # Convert the first tensor to a PIL image to get the width and height
    img_width, img_height = to_pil_image(batch[0]).size

    # Calculate the number of rows needed
    rows = len(batch) // width
    if len(batch) % width != 0:
        rows += 1

    # Create a new image of the appropriate size
    collage_width = width * img_width
    collage_height = rows * img_height
    collage = Image.new('RGB', (collage_width, collage_height))

    # Paste images into the collage
    for i, img_tensor in enumerate(batch):
        # Convert tensor to PIL image
        img = to_pil_image(img_tensor)

        # Calculate position
        x = (i % width) * img_width
        y = (i // width) * img_height

        # Paste the image
        collage.paste(img, (x, y))

    return collage



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="C:\\Users\Bbjar\Desktop\Kandidatsemester1\MLOps\some_data\lfw", type=str)
    parser.add_argument("-batch_size", default=512, type=int)
    parser.add_argument("-num_workers", default=1, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=100, type=int)

    args = parser.parse_args()
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
        ])
    
    lfw_trans = transforms.Compose([transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()])

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, transform = lfw_trans)

    # Define dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) 
    data_iterator = iter(dataloader)
    print(len(dataset))
    batch = next(data_iterator)

    # Assuming 'batch' is your batch of images and 'data_iterator' is your DataLoader iterator
    if args.visualize_batch:
        batch = next(data_iterator)
        collage = make_collage(batch, width=16)
        collage.show()

    if args.get_timing:
        workers = range(1, 6)
        mean_times = []
        std_devs = []
        
        for n in workers:
            res = []
            for _ in range(3):
                start = time.time()
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=n, pin_memory=True)
                for batch_idx, _batch in enumerate(dataloader):
                    if batch_idx >= args.batches_to_check:
                        break
                end = time.time()
                res.append(end - start)

            res = np.array(res)
            mean_times.append(np.mean(res))
            std_devs.append(np.std(res))

        plt.errorbar(workers, mean_times, yerr=std_devs, fmt='-o')
        plt.xlabel('Number of Workers')
        plt.ylabel('Average Time (seconds)')
        plt.title('Average Time vs Number of Workers')
        plt.savefig('timing_plot.png')

