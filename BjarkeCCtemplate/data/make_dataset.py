import torch
import os

def normalize_tensor(tensor):
    # Normalize the tensor to have mean 0 and std 1
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

def process_data(raw_dir, processed_dir):
    # Load and process training data
    train_images_list = []
    train_targets_list = []
    
    for i in range(6):  
        train_images_path = os.path.join(raw_dir, f'train_images_{i}.pt')
        train_targets_path = os.path.join(raw_dir, f'train_target_{i}.pt')
        
        train_images_batch = torch.load(train_images_path)
        train_targets_batch = torch.load(train_targets_path)
        
        train_images_list.append(train_images_batch)
        train_targets_list.append(train_targets_batch)

    # Combine the training batches into a single tensor
    all_train_images = torch.cat(train_images_list)
    all_train_targets = torch.cat(train_targets_list)

    # Normalize the training images tensor
    normalized_train_images = normalize_tensor(all_train_images)
   
    # Load and process test data
    test_images_path = os.path.join(raw_dir, 'test_images.pt')
    test_targets_path = os.path.join(raw_dir, 'test_target.pt')
    
    test_images = torch.load(test_images_path)
    test_targets = torch.load(test_targets_path)

    # Normalize the test images tensor
    normalized_test_images = normalize_tensor(test_images)
    
    # Save the processed tensors
    torch.save(normalized_train_images, os.path.join(processed_dir, 'processed_train_images.pt'))
    torch.save(all_train_targets, os.path.join(processed_dir, 'train_targets.pt'))
    torch.save(normalized_test_images, os.path.join(processed_dir, 'processed_test_images.pt'))
    torch.save(test_targets, os.path.join(processed_dir, 'test_targets.pt'))

if __name__ == '__main__':
    raw_data_dir = 'data/raw'  # Update this path to the correct location of your raw data
    processed_data_dir = 'data/processed'   # Update this path to where you want to save your processed data
    
    process_data(raw_data_dir, processed_data_dir)
    

