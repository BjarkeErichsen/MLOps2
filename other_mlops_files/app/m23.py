

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
torch.manual_seed(123)


def preprocess_image(image):
    """
    Preprocess the image for ResNet-152 model.

    Args:
    image (PIL.Image): Image to be preprocessed.

    Returns:
    Tensor: Preprocessed image tensor.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor



def predict_top5_classes(scripted_model, image_tensor):
    """
    Predict the top 5 categories of the image using the scripted model.

    Args:
    scripted_model (torch.jit.ScriptModule): The scripted ResNet-152 model.
    image_tensor (Tensor): The preprocessed image tensor.

    Returns:
    Tensor: Indices of the top 5 predicted categories.
    Tensor: Probabilities of the top 5 predicted categories.
    """
    with torch.no_grad():
        scripted_model.eval()
        outputs = scripted_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probabilities, top_categories = torch.topk(probabilities, 5)
        return top_categories[0], top_probabilities[0]


random_image = Image.fromarray(torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy())
image_tensor = preprocess_image(random_image)

# Download the model
model = models.resnet152(pretrained=True)
scripted_model = torch.jit.script(model)

# Predict the image
scripted_model_top_categories, top_probabilities = predict_top5_classes(scripted_model, image_tensor)
top_categories, top_probabilities = predict_top5_classes(model, image_tensor)
print("they are the same as can be seen on their outputs: ", top_categories, scripted_model_top_categories)



"""
A safer way to benchmark inference time
"""
import torch.utils.benchmark as benchmark

timer1 = benchmark.Timer(
    stmt='model(image_tensor)',
    setup='from __main__ import model, image_tensor',
    globals={'model': model, 'image_tensor': image_tensor},
    num_threads=torch.get_num_threads(),
    label="Inference Benchmark",
    sub_label="ResNet-152",
    description="Test on random data",
)

timer2 = benchmark.Timer(
    stmt='scripted_model(image_tensor)',
    setup='from __main__ import scripted_model, image_tensor',
    globals={'scripted_model': scripted_model, 'image_tensor': image_tensor},
    num_threads=torch.get_num_threads(),
    label="Inference Benchmark",
    sub_label="ResNet-152",
    description="Test on random data",
)

tim1 = timer1.blocked_autorange(min_run_time=1)

tim2 = timer2.blocked_autorange(min_run_time=1)

print(tim1, tim2)