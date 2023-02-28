from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class ImageData(Dataset):
    def __init__(self, root_path=None, name='dut'):
        if name == 'xu':
            self.source_path = root_path + 'CUHK/source/'
            self.dbd_path = root_path + 'CUHK/gt'
            self.form = '.bmp'
        if name == 'dut':
            self.source_path = root_path + 'DUT/source/'
            self.dbd_path = root_path + 'DUT/gt/'
            self.form = '.bmp'

        self.transform = {
            'test': transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
            ]),
        }
        self.transform_resize = {
            'test': transforms.Compose([
                transforms.ToTensor(),
            ]),
        }

    def __len__(self):
        """
        Get the length of the entire dataset
        """
        count = 0
        for fn in os.listdir(self.source_path):
            count = count + 1
        print("Length of dataset is ", count)
        return count

    def __getitem__(self, idx):
        """
        Get the image item by index
        """
        image_name = os.path.join(self.source_path, str(idx + 1) + '.bmp')
        image = Image.open(image_name)
        dbd_name = os.path.join(self.dbd_path, str(idx + 1) + '.bmp')
        dbd = Image.open(dbd_name)
        transformed_img = self.transform['test'](image)
        transformed_dbd = self.transform['test'](dbd)
        sample = {'image': transformed_img, 'dbd': transformed_dbd}
        return sample