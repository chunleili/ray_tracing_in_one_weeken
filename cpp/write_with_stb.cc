#include "color.h"
#include "vec3.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>

void save_img(uint8_t * data) // Saves image
{
    const int image_width = 256;
    const int image_height = 256;
    stbi_write_png("output.png", image_width, image_height, 3, data, image_width * 3);
}


int main() {
    const int image_width = 256;
    const int image_height = 256;
    uint8_t image_data[image_width * image_height * 3];
    
    int k = 0;
    for (int j = image_height-1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            auto r = double(i) / (image_width-1);
            auto g = double(j) / (image_height-1);
            auto b = 0.25;

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            // std::cout << ir << ' ' << ig << ' ' << ib << '\n';

            image_data[k++] = ir;
            image_data[k++] = ig;
            image_data[k++] = ib;
        }
    }

    save_img(image_data);

    return 0;
}