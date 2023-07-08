import os, shutil

# # 删除Build目录
# shutil.rmtree('build')

# # 编译
os.system("cmake -B build")
os.system("cmake --build build --config Debug")
# 运行
os.system(".\\build\\Debug\\main.exe > img.ppm")