import skimage.measure
import numpy as np
import imageio
from PIL import Image
import skimage.transform

img = Image.open("C:/Users/67059/Desktop/1.png")

# 获取图像的宽度和高度
width, height = img.size

# 计算中心点
center = width // 2

# 分割图像
left_half = img.crop((0, 0, center, height))
right_half = img.crop((center, 0, width, height))
left_half.save("left_half.png")
right_half.save("right_half.png")

x_real = imageio.imread('left_half.png', as_gray=True)
x_fake = imageio.imread('right_half.png', as_gray=True)

# 调整输入图像的大小
if x_real.shape != x_fake.shape:
    max_shape = np.maximum(x_real.shape, x_fake.shape)
    x_real = skimage.transform.resize(x_real, max_shape)
    x_fake = skimage.transform.resize(x_fake, max_shape)

ssim_res = skimage.measure.compare_ssim(x_real, x_fake)
print(ssim_res)