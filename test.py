import os.path as osp
import random
from tqdm import tqdm
import cv2
import numpy as np
import paddle


from utils.utils import load_img, save_img
files = "/home/data/disk2/wsq/code/SRMNet_Paddle-main-master/pic/0080.png"
result_dir_tmp = "/home/data/disk2/wsq/code/SRMNet_Paddle-main-master/pic_noise/1.png"

img = np.float32(load_img(files)) / 255.

np.random.seed(seed=0)  # for reproducibility
sigma_value = random.uniform(0, 50)

noise_level = sigma_value / 255.0
# noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
# noise = np.random.randn(*img_lq.shape) * noise_level
noise = paddle.randn(img.shape,dtype='float32').numpy() * noise_level

img += noise

img = img * 255.0

save_img(result_dir_tmp, img)

# with paddle.no_grad():
#         for file_ in tqdm(files):
#             img = np.float32(load_img(file_)) / 255.

#             np.random.seed(seed=0)  # for reproducibility
#             sigma_value = random.uniform(0, 50)

#             noise_level = sigma_value / 255.0
#             # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
#             # noise = np.random.randn(*img_lq.shape) * noise_level
#             noise = paddle.randn(img.shape,dtype='float32').numpy() * noise_level
            
#             img += noise

#             # img = paddle.to_tensor(img)
#             # img = paddle.transpose(img, [2, 0, 1])
#             # input_ = img.unsqueeze(0)

#             # # Padding in case images are not multiples of 8
#             # h, w = input_.shape[2], input_.shape[3]
#             # H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
#             # padh = H - h if h % factor != 0 else 0
#             # padw = W - w if w % factor != 0 else 0
#             # input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

#             # restored = model_restoration(input_)

#             # # Unpad images to original dimensions
#             # restored = restored[:, :, :h, :w]

#             # restored = paddle.clip(restored, 0, 1).detach()
#             # restored = paddle.transpose(restored, [0, 2, 3, 1]).squeeze(0).numpy()

#             save_file = os.path.join(result_dir_tmp, "noise_" + os.path.split(file_)[-1])
#             save_img(save_file, img)

#             # img = paddle.transpose(img, [1, 2, 0]).clip(0, 1).numpy()
#             # save_file = os.path.join(result_dir_tmp, "noise_" + os.path.split(file_)[-1])
#             # save_img(save_file, img_as_ubyte(img))
# print(f"The predict image save in {result_dir_tmp} path.")