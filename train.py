# Train a new network on a data set with train.py
#
# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
#   * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#   * Choose architecture: python train.py data_dir --arch "vgg13"
#   * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#   * Use GPU for training: python train.py data_dir --gpu
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str,
                        help='path to the image folder')
    parser.add_argument('save_dir', type=str,
                        help='path to where the model shall be saved', default=".")
    parser.add_argument('--arch', type=str, default='vgg11',
                        help='CNN Model Architecture (pick any from https://pytorch.org/vision/stable/models.html#classification)')
    parser.add_argument('--epochs', type=int, default=20, action='store_true',
                        help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Hidden units')
    parser.add_argument('--gpu', default=True, action='store_true',
                        help='Use GPU?')
    return parser.parse_args()


def select_model(model, hidden_units):
    selected_model = models.get_model(model, weights="DEFAULT")
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return selected_model


def train_model(model, dir, epochs, learning_rate, use_gpu):
    train_dataloaders = datasets.ImageFolder(dir, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    test_dataloaders = datasets.ImageFolder(dir, transform=transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        for inputs, labels in train_dataloaders:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_dataloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(test_dataloaders):.3f}.. "
                      f"Test accuracy: {accuracy / len(test_dataloaders):.3f}")
                running_loss = 0
                model.train()
    return model, train_dataloaders.class_to_idx, optimizer


def save_model(model, class_to_idx, epochs, optimizer, arch):
    torch.save({
        'class_to_idx': class_to_idx,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'epochs': epochs,
        'arch': arch
    }, 'model.pth')


def main():
    in_arg = get_input_args()
    data_dir = in_arg.dir
    save_dir = in_arg.save_dir
    epochs = in_arg.epochs
    learning_rate = in_arg.learning_rate
    hidden_units = in_arg.hidden_units
    arch = in_arg.arch
    useGpu = in_arg.gpu

    model = select_model(arch, hidden_units)
    model, class_to_idx, optimizer = train_model(model, data_dir, epochs, learning_rate, useGpu)
    save_model(model, class_to_idx, epochs, optimizer, arch)


# Call to main function to run the program
if __name__ == "__main__":
    main()
