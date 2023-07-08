import numpy as np
import taichi as ti
ti.init()

# 假设图像数据是一个二维数组，表示RGB图像
width = 256  # 图像宽度
height = 256  # 图像高度
channels = 3  # RGB图像，通道数为3

image_data = np.zeros(dtype=np.uint8, shape =(width*height, channels))
image_index = ti.field(dtype=ti.i32, shape=())
@ti.func
def write_color_to_data(pixel_color:ti.math.vec3, image_data:ti.types.ndarray()):
    k = ti.atomic_add(image_index[None], 1)
    image_data[k, 0] = int(255.999 * pixel_color[0])
    image_data[k, 1] = int(255.999 * pixel_color[1])
    image_data[k, 2] = int(255.999 * pixel_color[2])
    
@ti.kernel
def paint_simple():
    ti.loop_config(serialize=True)
    for j in range(width):
        for i in range(height):
            r = i / (width-1)
            g = j / (height-1)
            b = 0.25

            ir = int(255.999 * r)
            ig = int(255.999 * g)
            ib = int(255.999 * b)

            print(ir, ig, ib)

@ti.kernel
def paint(image_data:ti.types.ndarray()):
    ti.loop_config(serialize=True)
    for j in range(width):
        for i in range(height):
            r = i / (width-1)
            g = j / (height-1)
            b = 0.25
            write_color_to_data(ti.math.vec3([r, g, b]), image_data)

paint(image_data)

#写出ppm
def write_ppm(image_data):
    with open('taichi.ppm', 'w') as f:
        f.write('P3\n{} {}\n255\n'.format(width, height))
        for i in range(width*height):
            f.write('{} {} {}\n'.format(image_data[i, 0], image_data[i, 1], image_data[i, 2]))

write_ppm(image_data)
