import numpy as np
import taichi as ti
ti.init()

# 假设图像数据是一个二维数组，表示RGB图像
aspect_ratio = 16.0 / 9.0
width = 400  # 图像宽度
height = int(width / aspect_ratio)  # 图像高度
channels = 3  # RGB图像，通道数为3

# camera
viewport_height = 2.0
viewport_width = aspect_ratio * viewport_height
focal_length = 1.0

origin = ti.math.vec3(0, 0, 0)
horizontal = ti.math.vec3(viewport_width, 0, 0)
vertical = ti.math.vec3(0, viewport_height, 0)
lower_left_corner = origin - horizontal/2 - vertical/2 - ti.math.vec3([0, 0, focal_length])


@ti.dataclass
class Ray:
    orig: ti.math.vec3
    dir: ti.math.vec3
    @ti.func
    def at(self, t):
        return self.orig + t*self.dir


@ti.func
def hit_sphere(center:ti.math.vec3, radius:float, r:Ray):
    oc = r.orig - center
    a = r.dir.dot(r.dir)
    b = 2.0 * oc.dot(r.dir)
    c = oc.dot(oc) - radius*radius
    discriminant = b*b - 4*a*c
    res = -1.0
    if (discriminant < 0.0):
        res = -1.0
    else:
        res = (-b - ti.sqrt(discriminant)) / (2.0*a)
    return res

@ti.func
def ray_color(r)->ti.math.vec3:
    t = hit_sphere(ti.math.vec3([0, 0, -1]), 0.5, r)
    res = ti.math.vec3([0, 0, 0])
    if (t > 0.0):
        N = r.at(t) - ti.math.vec3([0, 0, -1])
        N = N.normalized()
        res = 0.5 * ti.math.vec3([N[0]+1, N[1]+1, N[2]+1])
    else:
        unit_direction = r.dir.normalized()
        t = 0.5 * (unit_direction[1] + 1.0)
        res = (1.0 - t) * ti.math.vec3([1.0, 1.0, 1.0]) + t * ti.math.vec3([0.5, 0.7, 1.0])
    return res

@ti.func
def write_color(pixel_color:ti.math.vec3):
    print(int(255.999 * pixel_color[0]), int(255.999 * pixel_color[1]), int(255.999 * pixel_color[2]))


image_data = np.zeros(dtype=np.uint8, shape =(width*height, channels))
image_index = ti.field(dtype=ti.i32, shape=())
@ti.func
def write_color_to_data(pixel_color:ti.math.vec3, image_data:ti.types.ndarray()):
    k = ti.atomic_add(image_index[None], 1)
    image_data[k, 0] = int(255.999 * pixel_color[0])
    image_data[k, 1] = int(255.999 * pixel_color[1])
    image_data[k, 2] = int(255.999 * pixel_color[2])

@ti.kernel
def paint(image_data:ti.types.ndarray()):
    ti.loop_config(serialize=True)
    for j in range(height):
        jj = height - 1 - j
        for i in range(width):
            u = i / (width-1)
            v = jj / (height-1)
            r = Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin)
            pixel_color = ray_color(r)
            write_color_to_data(pixel_color, image_data)
            # write_color(pixel_color)

paint(image_data)

print("done")
#写出ppm
def write_ppm(image_data):
    with open('taichi.ppm', 'w') as f:
        f.write('P3\n{} {}\n255\n'.format(width, height))
        for i in range(width*height):
            f.write('{} {} {}\n'.format(image_data[i, 0], image_data[i, 1], image_data[i, 2]))
write_ppm(image_data)