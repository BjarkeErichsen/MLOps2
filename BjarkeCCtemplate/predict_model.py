import click
import torch
from torch import nn
from models.model import myawesomemodel
import matplotlib.pyplot as plt
from pathlib import Path


@click.group()
def cli():
    """Simple CLI calculator."""
    pass


def processed_mnist():
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [], []

    # train_data.append(torch.load(f"data/processed/train_images.pt"))
    # train_labels.append(torch.load(f"data/processed/train_targets.pt"))
    working_dir = Path("data/processed")
    test_images = working_dir / "test_images.pt"
    test_labels = working_dir / "test_targets.pt"

    test_data = torch.load(test_images)
    test_labels = torch.load(test_labels)

    print(test_data.shape)
    print(test_labels.shape)

    test_data = test_data.unsqueeze(1)

    return torch.utils.data.TensorDataset(test_data, test_labels)


@click.command()
@click.argument(model_path, nargs=-1, type=str)
@click.argument(batch_size, default=10, type=int)
def predict(model_path, batch_size) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = myawesomemodel.to(device)
    test_set = processed_mnist()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    return torch.cat([model(batch) for batch in test_dataloader], 0)


cli.add_command(predict)

if __name__ == "__main__":
    cli()
