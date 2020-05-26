import torch
from FashionMnistCNN.ConvolutionalNeuralNetwork import CNN
from torchvision import transforms


class ImageProcessor:
    def __init__(self, model_path):
        self.device = self.get_device()
        with torch.no_grad():
            cnn = self.load_cnn_from_file(model_path)
            self.model = cnn.to(self.device)

    def process_image(self, image):
        batch_t = self.transform_image(image)

        with torch.no_grad():
            input = batch_t.to(self.device)
            output = self.model(input)

        output = output.cpu()
        result = self.get_class(output.argmax().item())
        return result

    def load_cnn_from_file(self, path):
        model = CNN()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def get_device(self):
        if torch.cuda.is_available():
            print('Using CUDA on: ' + torch.cuda.get_device_name(0))
        else:
            print('CUDA not available, using CPU')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device

    def transform_image(self, image):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_t = transform(image)
        batch_t = torch.unsqueeze(img_t, 0)
        return batch_t

    def get_class(self, class_id):
        return {
            0: "a t-shirt/top",
            1: "a pair of trousers",
            2: "a pullover",
            3: "a dress",
            4: "a coat",
            5: "a sandal",
            6: "a shirt",
            7: "a sneaker",
            8: "a bag",
            9: "an ankle boot",
        }[class_id]
