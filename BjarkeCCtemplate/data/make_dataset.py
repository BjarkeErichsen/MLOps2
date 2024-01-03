if __name__ == "__main__":
    import os
    import torch

    def load_tensors_from_folder(folder):
        tensors = []
        for filename in os.listdir(folder):
            if filename.endswith('.pt'):  # Ensuring to process only .pt files
                tensor_path = os.path.join(folder, filename)
                tensor = torch.load(tensor_path)
                tensors.append(tensor)
        return tensors

    def normalize(tensor):
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std

    # Define the relative paths
    script_dir = os.path.dirname(__file__)
    raw_data_folder = os.path.join(script_dir, '..\..\data\raw\corruptmnist')
    processed_folder_path = os.path.join(script_dir, 'data\processed')

    # Load, process, and save tensors
    tensors = load_tensors_from_folder(raw_data_folder)
    normalized_tensors = [normalize(tensor) for tensor in tensors]

    # Ensure the processed folder exists
    if not os.path.exists(processed_folder_path):
        os.makedirs(processed_folder_path)

    # Save each tensor
    for i, tensor in enumerate(normalized_tensors):
        torch.save(tensor, os.path.join(processed_folder_path, f'processed_tensor_{i}.pt'))
