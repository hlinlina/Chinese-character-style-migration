import tensorflow.compat.v1 as tf
from PIL import Image

img = Image.open("C:/Users/67059/Desktop/3.png")

# 获取图像的宽度和高度
width, height = img.size

# 计算中心点
center = width // 2

# 分割图像
left_half = img.crop((0, 0, center, height))
right_half = img.crop((center, 0, width, height))

# 调整图像大小为相同尺寸
left_half = left_half.resize(right_half.size)

left_half.save("left_half.png")
right_half.save("right_half.png")

def read_img(path):
    return tf.image.decode_image(tf.io.read_file(path))

def psnr(tf_img1,tf_img2):
    psnr_val = tf.image.psnr(tf_img1,tf_img2,max_val=255)
    tf.print("PSNR:", psnr_val)
    return psnr_val

@tf.function
def _main():
    t1=read_img('left_half.png')
    t2=read_img('right_half.png')
    y = psnr(t1,t2)

if __name__=='__main__':
    _main()