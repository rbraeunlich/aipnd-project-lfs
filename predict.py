import argparse
import torch
from torch import nn
from torchvision import models
import numpy as np
from PIL import Image
from collections import OrderedDict
import json


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_image', type=str, help='path to the image file')
    parser.add_argument('checkpoint', type=str,
                        help='path to where the model checkpoint has been saved')
    parser.add_argument('--top_k', type=int, default=5, help='Top K classes')
    parser.add_argument('--category_names', type=str, help='mapping to category names')
    parser.add_argument('--gpu', default=True, action='store_true',
                        help='Use GPU?')
    return parser.parse_args()


def load_model(checkpoint):
    model_checkpoint = torch.load(checkpoint)
    model = models.get_model(model_checkpoint['arch'], weights="DEFAULT")
    hidden_units = model_checkpoint['hidden_units']
    num_ftrs = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_ftrs, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.load_state_dict(model_checkpoint['model_state_dict'])
    class_to_idx = model_checkpoint['class_to_idx']
    return model, class_to_idx


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image_path) as image:
        # Resize the image where shortest side is 256 pixels
        width, height = image.size
        aspect_ratio = width / height
        if width < height:
            new_width = 256
            new_height = int(new_width / aspect_ratio)
        elif width > height:
            new_height = 256
            new_width = int(new_height * aspect_ratio)
        else:  # when both sides are equal
            new_width = 256
            new_height = 256

        image = image.resize((new_width, new_height))

        # Crop the center 224x224 portion
        left_margin = (new_width - 224) / 2
        bottom_margin = (new_height - 224) / 2
        right_margin = left_margin + 224
        top_margin = bottom_margin + 224

        image = image.crop((left_margin, bottom_margin, right_margin, top_margin))

        # Normalize
        np_image = np.array(image) / 255  # Scale pixel values
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std  # Normalize

        # Reorder dimensions
        np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(image_path, model, class_to_idx, use_gpu, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model = model.eval()
    img = process_image(image_path)
    probabilities = model(torch.from_numpy(img).unsqueeze(0).float().to(device))
    probs, top_class = probabilities.topk(topk, dim=1)
    inv_dict = dict(map(reversed, class_to_idx.items()))
    classes = [inv_dict[i.item()] for i in top_class.flatten()]
    return torch.exp(probs), classes


def main():
    in_arg = get_input_args()
    path_to_image = in_arg.path_to_image
    checkpoint = in_arg.checkpoint
    top_k = in_arg.top_k
    category_file = in_arg.category_names
    use_gpu = in_arg.gpu

    model, class_to_idx = load_model(checkpoint)
    probs, classes = predict(path_to_image, model, class_to_idx, use_gpu, top_k)

    if category_file:
        with open(category_file, 'r') as f:
            cat_to_name = json.load(f)
            categories = [cat_to_name[i] for i in classes]
            # convert tensor to numpy array and flatten it
            probs_np = probs.detach().cpu().flatten().numpy()
            # create pairs of categories and probabilities
            pairs = list(zip(categories, probs_np))
            # sort pairs in descending order of probabilities
            pairs.sort(key=lambda x: x[1])
            # separate sorted categories and probabilities
            categories_sorted, probs_sorted = zip(*pairs)
            print(categories_sorted)
            print(probs_sorted)
            return categories_sorted, probs_sorted
    else:
        print(classes)
        print(probs)
        return classes, probs


if __name__ == "__main__":
    main()
