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
from torchvision import datasets, transforms, models

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str,
                        help='path to the image folder')
    parser.add_argument('save_dir', type=str,
                        help='path to where the model shall be saved', default=".")
    parser.add_argument('--arch', type=str, default='vgg', choices=['vgg', 'alexnet', 'resnet'],
                        help='CNN Model Architecture (resnet, alexnet, or vgg)')
    parser.add_argument('--epochs', type=int, default=20, action='store_true',
                        help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Hidden units')
    parser.add_argument('--gpu', default=True, action='store_true',
                        help='Use GPU?')
    return parser.parse_args()


def select_model(model):
    return models.get_model(model, weights="DEFAULT")


def train_model(model, dir, epochs, learning_rate, hidden_units, useGpu):
    train_image_dataset = datasets.ImageFolder(dir, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    pass


def main():
    in_arg = get_input_args()
    dir = in_arg.dir
    save_dir = in_arg.save_dir
    epochs = in_arg.epochs
    learning_rate = in_arg.learning_rate
    hidden_units = in_arg.hidden_units
    arch = in_arg.arch
    useGpu = in_arg.gpu

    model = select_model(arch)
    train_model(model, dir, epochs, learning_rate, hidden_units, useGpu)


# Call to main function to run the program
if __name__ == "__main__":
    main()
