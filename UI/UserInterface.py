from threading import Thread
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

from FashionMnistCNN.ImageProcessor import ImageProcessor


class UserInterface:
    def __init__(self, root, image_processor):
        self.root = root
        self.button = self.create_open_image_button()
        self.image_panel = self.create_image()
        self.image_label = self.create_label()
        self.image_processor = image_processor

    def create_open_image_button(self):
        button = Button(self.root, text='Open image', command=self.process_image)
        button.pack(padx=10, pady=5)
        return button

    def create_image(self):
        image = Label(self.root, image=None)
        image.pack(padx=10, pady=5)
        return image

    def create_label(self):
        label = Label(self.root, text='', font=("Helvetica", 16))
        label.pack(padx=10, pady=5)
        return label

    def mainloop(self):
        self.root.mainloop()

    def process_image(self):
        image = self.load_image()
        self.image_label.configure(text='Recognizing...')
        Thread(target=self.recognize_image, args=(image,)).start()

    def load_image(self):
        filename = filedialog.askopenfilename(title='Open image')
        image = Image.open(filename)
        image_to_display = self.resize_image(image)
        self.image_panel.configure(image=image_to_display)
        self.image_panel.image = image_to_display
        return image

    def recognize_image(self, image):
        label = "This is " + image_processor.process_image(image)
        self.image_label.configure(text=label)

    def resize_image(self, image):
        image = image.resize((252, 252), Image.NEAREST)
        image = ImageTk.PhotoImage(image)
        return image


root = Tk()
root.title('Fashion MNIST CNN')
root.geometry("280x350+600+300")
root.resizable(width=True, height=True)

image_processor = ImageProcessor('../Data/model.pth')

user_interface = UserInterface(root, image_processor)
user_interface.mainloop()
