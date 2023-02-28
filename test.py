import os
import torch
from data_test import *
from ResNet_FillAndHollow import *
from torch.utils.data import DataLoader
from torchvision import transforms

model_name = "resnet_fill_hollow"
result_path = "./result/"
model_path = "./checkpoint/"
test_path = "./dataset/test"

dbd_dataset_dut = ImageData(root_path=test_path, name='dut')
dbd_dataset_xu = ImageData(root_path=test_path, name='xu')

model = resnet_dbd_edge()

model = model.cuda()

net = torch.load(model_path+'checkpoint.pth')

model_new = {}
for k, v in net.items():
    k = k.split('module.')[-1]
    model_new[k] = v
model.load_state_dict(model_new)
model.eval()

for k in range(1):
    if k == 0:
        name = 'xu'
        dataloader = DataLoader(dbd_dataset_xu, batch_size=1, shuffle=False)
    if  k == 1:
        name = 'dut'
        dataloader = DataLoader(dbd_dataset_dut, batch_size=1, shuffle=False)

    count_yes = 0
    count_no = 0
    save_path = result_path + 'result'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, sample_batch in enumerate(dataloader):
        images_batch, dbd_batch = sample_batch[0]['image'], sample_batch[0]['dbd']
        if torch.cuda.is_available():
            input_image = Variable(images_batch.cuda())
            dbd = Variable(dbd_batch.cuda())
        else:
            input_image = Variable(images_batch)
            dbd = Variable(dbd_batch)

        output1_dbd1, _  = model(input_image)
        output_dbd1 = output1_dbd1.cpu()
        output_dbd1 = output_dbd1[0, :, :, :]
        output_dbd1 = torch.squeeze(output_dbd1)
        img = transforms.ToPILImage()(output_dbd1)
        img.save(os.path.join(save_path, str(i+1)  + '.bmp'))