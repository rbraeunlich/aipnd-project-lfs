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
    parser.add_argument('--save_dir', type=str,
                        help='path to where the model shall be saved', default="your-model.pth")
    parser.add_argument('--arch', type = str, default = 'vgg', choices = ['vgg', 'alexnet', 'resnet'],
                        help = 'CNN Model Architecture (resnet, alexnet, or vgg)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Hidden units')
    parser.add_argument('--gpu', default=True, action='store_true',
                        help='Use GPU?')
    return parser.parse_args()


def select_model(model, hidden_units):
    selected_model = models.vgg16(weights="DEFAULT")
    num_ftrs = 25088
    if model == 'alexnet':
        selected_model = models.alexnet(weights="DEFAULT")
        num_ftrs = 9216
    elif model == 'resnet':
        selected_model = models.resnet18(weights="DEFAULT")
        num_ftrs = 512
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_ftrs, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    selected_model.classifier = classifier

    return selected_model


def load_images(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=data_transforms)
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=32, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=32)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=32)
    return train_dataloaders, valid_dataloaders, test_dataloaders, train_image_datasets


def validate_model(model, valid_dataloaders, device):
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in valid_dataloaders:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Validation accuracy: {accuracy/len(valid_dataloaders):.3f}")


def train_model(model, dir, epochs, learning_rate, use_gpu):
    train_dataloaders, valid_dataloaders, test_dataloaders, train_image_datasets = load_images(dir)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
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
    validate_model(model, valid_dataloaders, device)
    return model, train_image_datasets.class_to_idx, optimizer


def save_model(model, class_to_idx, arch, hidden_units, save_dir):
    torch.save({
        'class_to_idx': class_to_idx,
        'model_state_dict': model.state_dict(),
        'hidden_units': hidden_units,
        'arch': arch
    }, save_dir)


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
    save_model(model, class_to_idx, arch, hidden_units, save_dir)


# Call to main function to run the program
if __name__ == "__main__":
    main()
