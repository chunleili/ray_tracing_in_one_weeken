import numpy as np
from dataclasses import dataclass

# 假设图像数据是一个二维数组，表示RGB图像
aspect_ratio = 16.0 / 9.0
width = 400  # 图像宽度
height = int(width / aspect_ratio)  # 图像高度
channels = 3  # RGB图像，通道数为3

# camera
viewport_height = 2.0
viewport_width = aspect_ratio * viewport_height
focal_length = 1.0

origin = np.array([0, 0, 0])
horizontal = np.array([viewport_width, 0, 0])
vertical = np.array([0, viewport_height, 0])
lower_left_corner = origin - horizontal/2 - vertical/2 - np.array([0, 0, focal_length])

@dataclass
class Ray:
    orig: np.ndarray
    dir: np.ndarray
    def at(self, t):
        return self.orig + t*self.dir


def hit_sphere(center:np.ndarray, radius:float, r:Ray):
    oc = r.orig - center
    a = r.dir.dot(r.dir)
    b = 2.0 * oc.dot(r.dir)
    c = oc.dot(oc) - radius*radius
    discriminant = b*b - 4*a*c
    res = -1.0
    if (discriminant < 0.0):
        res = -1.0
    else:
        res = (-b - np.sqrt(discriminant)) / (2.0*a)
    return res

def ray_color(r)->np.ndarray:
    t = hit_sphere(np.array([0, 0, -1]), 0.5, r)
    res = np.array([0, 0, 0])
    if (t > 0.0):
        N = r.at(t) - np.array([0, 0, -1])
        N = N / np.linalg.norm(N)
        res = 0.5 * np.array([N[0]+1, N[1]+1, N[2]+1])
    else:
        unit_direction = r.dir/np.linalg.norm(r.dir)
        t = 0.5 * (unit_direction[1] + 1.0)
        res = (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])
    return res

def write_color(pixel_color:np.ndarray):
    print(int(255.999 * pixel_color[0]), int(255.999 * pixel_color[1]), int(255.999 * pixel_color[2]))


def write_color_to_data(pixel_color:np.ndarray, image_data:np.ndarray, image_index:int):
    image_data[image_index, 0] = int(255.999 * pixel_color[0])
    image_data[image_index, 1] = int(255.999 * pixel_color[1])
    image_data[image_index, 2] = int(255.999 * pixel_color[2])

def paint(image_data:np.ndarray):
    image_index = 0
    for j in range(height):
        jj = height - 1 - j
        for i in range(width):
            u = i / (width-1)
            v = jj / (height-1)
            r = Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin)
            pixel_color = ray_color(r)
            write_color_to_data(pixel_color, image_data, image_index)
            image_index+=1
            # write_color(pixel_color)
    
#写出ppm
def write_ppm(image_data):
    with open('taichi.ppm', 'w') as f:
        f.write('P3\n{} {}\n255\n'.format(width, height))
        for i in range(width*height):
            f.write('{} {} {}\n'.format(image_data[i, 0], image_data[i, 1], image_data[i, 2]))

def main():
    image_data = np.zeros(dtype=np.uint8, shape =(width*height, channels))
    paint(image_data)
    print("done")
    write_ppm(image_data)

if __name__ == "__main__":
    main()