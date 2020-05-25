import torch
from FashionMnistCNN.ConvolutionalNeuralNetwork import CNN
from torchvision import transforms


def process_image(image):

    batch_t = transform_image(image)
    model = load_cnn_from_file('../Data/model.pth')
    device = get_device()

    with torch.no_grad():
        model = model.to(device)
        input = batch_t.to(device)
        output = model(input)

    output = output.cpu()
    result = get_class(output.argmax().item())
    return result


def load_cnn_from_file(path):
    model = CNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def get_device():
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print('CUDA not available, using CPU')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def transform_image(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t


def get_class(class_id):
    return {
        0: "t-shirt/top",
        1: "a pair of trousers",
        2: "pullover",
        3: "dress",
        4: "coat",
        5: "sandal",
        6: "shirt",
        7: "sneaker",
        8: "bag",
        9: "ankle boot",
    }[class_id]


