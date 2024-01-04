import click
import torch
from torch import nn
from models.model import myawesomemodel 
import matplotlib.pyplot as plt

def processed_mnist():
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [ ], [ ]

    train_data.append(torch.load(f"data\processed\processed_test_images.pt\processed_train_images.pt"))
    train_labels.append(torch.load(f"data\processed\processed_test_images.pt\train_targets.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load("dataprocessed\processed_test_images.pt")
    test_labels = torch.load("data\processed\processed_test_targets.pt")

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    return (
        torch.utils.data.TensorDataset(train_data, train_labels), 
        torch.utils.data.TensorDataset(test_data, test_labels)
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--num_epochs", default=20, help="number of epochs to train for")
def train(lr, batch_size, num_epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(batch_size)

    # TODO: Implement training loop here
    model = myawesomemodel.to(device)
    train_set, _ = processed_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    epoch_losses = []   
    for epoch in range(num_epochs):
        batch_losses = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        print(f"Epoch {epoch} Loss {loss}")

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_loss)

    torch.save(model, f"models/model{lr}_{batch_size}_{num_epochs}.pt")
 
     # Save the training curve
    plt.figure()
    plt.plot(epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)

    # Make sure the directory exists
    figures_directory = "reports/figures/"

    # Save the plot
    training_curve_path = f"{figures_directory}training_curve_lr{lr}_bs{batch_size}_epochs{num_epochs}.png"
    plt.savefig(training_curve_path)
    print(f"Training curve saved to {training_curve_path}")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = processed_mnist()
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=False
    )
    model.eval()

    test_preds = [ ]
    test_labels = [ ]
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            test_preds.append(y_pred.argmax(dim=1).cpu())
            test_labels.append(y.cpu())

    test_preds = torch.cat(test_preds, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    print((test_preds == test_labels).float().mean())


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()