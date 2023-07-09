import numpy as np
from abc import abstractmethod
from dataclasses import dataclass

aspect_ratio = 16.0 / 9.0
width = 400  # 图像宽度
height = int(width / aspect_ratio)  # 图像高度
channels = 3  # RGB图像，通道数为3


class Camera:
    def __init__(self):
        aspect_ratio = 16.0 / 9.0
        viewport_height = 2.0
        viewport_width = aspect_ratio * viewport_height
        focal_length = 1.0

        self.origin = np.array([0., 0, 0])
        self.horizontal = np.array([viewport_width, 0, 0])
        self.vertical = np.array([0., viewport_height, 0])
        self.lower_left_corner = self.origin - self.horizontal/2 - self.vertical/2 - np.array([0, 0, focal_length])

    def get_ray(self, u, v):
        return Ray(self.origin, self.lower_left_corner + u*self.horizontal + v*self.vertical - self.origin)
        

@dataclass
class HitRecord:
    p: np.array
    normal: np.array
    t: float

class Hittable:
    @abstractmethod
    def hit(self, r, t_min, t_max, rec):
        pass


class Sphere(Hittable):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def hit(self, r, t_min, t_max, rec)->bool:
        oc = r.orig - self.center
        a = r.dir.dot(r.dir)
        half_b = oc.dot(r.dir)
        c = np.linalg.norm(oc)**2 - self.radius*self.radius

        discriminant = half_b*half_b - a*c
        res = True
        if (discriminant < 0):
            res = False
        else:
            sqrtd = np.sqrt(discriminant)

            # Find the nearest root that lies in the acceptable range.
            root = (-half_b - sqrtd) / a
            if (root < t_min or t_max < root):
                root = (-half_b + sqrtd) / a
                if (root < t_min or t_max < root):
                    res = False
            else:
                rec.t = root
                rec.p = r.at(rec.t)
                outward_normal = (rec.p - self.center) / self.radius
                rec.set_face_normal(r, outward_normal)
                res = True
        return res


@dataclass
class HitRecord:
    p: np.array
    normal: np.array
    t: float
    front_face: bool
    def set_face_normal(self, r, outward_normal):
        self.front_face = r.dir.dot(outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal
        ...


class HittableList(Hittable):
    def __init__(self):
        self.objects = []

    def clear(self):
        self.objects.clear()

    def add(self, object):
        self.objects.append(object)

    def hit(self, r, t_min, t_max, rec)->bool:
        hit_anything = False
        closest_so_far = t_max

        for object in self.objects:
            if object.hit(r, t_min, closest_so_far, rec):
                hit_anything = True
                closest_so_far = rec.t
        return hit_anything

@dataclass
class Ray:
    orig: np.array
    dir: np.array
    def at(self, t):
        return self.orig + t*self.dir

def ray_color(r, world):
    rec = HitRecord(np.array([0, 0, 0]), np.array([0, 0, 0]), 0.0, False)
    res = np.array([0, 0, 0])
    if world.hit(r, 0, float('inf'), rec):
        res = 0.5 * (rec.normal + np.array([1, 1, 1]))
    else:
        unit_direction = r.dir / np.linalg.norm(r.dir)
        t = 0.5 * (unit_direction[1] + 1.0)
        res = (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])
    return res


def write_color(pixel_color:np.ndarray):
    print(int(255.999 * pixel_color[0]), int(255.999 * pixel_color[1]), int(255.999 * pixel_color[2]))


def write_color_to_data(pixel_color:np.ndarray, image_data:np.ndarray, image_index:int):
    image_data[image_index, 0] = int(255.999 * pixel_color[0])
    image_data[image_index, 1] = int(255.999 * pixel_color[1])
    image_data[image_index, 2] = int(255.999 * pixel_color[2])

def paint(image_data:np.ndarray, world:HittableList, cam:Camera):
    image_index = 0
    for j in range(height):
        jj = height - 1 - j
        for i in range(width):
            u = i / (width-1)
            v = jj / (height-1)
            r = cam.get_ray(u, v)
            pixel_color = ray_color(r, world)
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
    world = HittableList()
    world.add(Sphere(np.array([0, 0, -1]), 0.5))
    world.add(Sphere(np.array([0, -100.5, -1]), 100))

    cam = Camera()

    image_data = np.zeros(dtype=np.uint8, shape =(width*height, channels))

    paint(image_data, world, cam)
    print("done")
    write_ppm(image_data)

if __name__ == '__main__':
    main()