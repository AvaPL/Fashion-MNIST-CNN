from tkinter import *
from tkinter import filedialog

from PIL import ImageTk, Image


class UserInterface:
    def __init__(self, root):
        self.root = root
        self.button = self.create_open_image_button()
        self.image_panel = self.create_image()
        self.image_label = self.create_label()

    def create_open_image_button(self):
        button = Button(self.root, text='Open image', command=self.set_image_and_label)
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

    def set_image_and_label(self):
        filename = filedialog.askopenfilename(title='Open image')
        image = self.open_image(filename)
        self.image_panel.configure(image=image)
        self.image_panel.image = image
        label = 'Hey, I\'m new here'  # TODO: Add image recognition here
        self.image_label.configure(text=label)

    def open_image(self, file):
        image = Image.open(file)
        image = image.resize((250, 250), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        return image


root = Tk()
root.title('Fashion MNIST CNN')
root.geometry("280x350+600+300")
root.resizable(width=True, height=True)

user_interface = UserInterface(root)
user_interface.mainloop()
