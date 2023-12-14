'''Need to make sure that this torch file works. Its on my todo list'''

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import requests
from io import BytesIO

# Function to load an image from a URL
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Function to predict the class of an image using ResNet50
def classify_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Define the transformation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Check if a GPU is available and if not, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained ResNet50 model
    model = resnet50(pretrained=True)
    model = model.to(device)
    model.eval()

    # Move the input and model to GPU for speed if available
    input_batch = input_batch.to(device)

    with torch.no_grad():
        # Model inference
        output = model(input_batch)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Load the labels
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    # Print top 5 categories
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(labels[top5_catid[i]], top5_prob[i].item())

# Example usage
image_path = "path_to_your_image.jpg"  # Replace with your image path
classify_image(image_path)
