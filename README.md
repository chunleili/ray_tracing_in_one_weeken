练习 ray tracing in one weekend

书本
https://raytracing.github.io/books/RayTracingInOneWeekend.html


环境：cmake3.25 win10 VS2022 taichi 1.6.0

- cpp： 原书cpp代码
- taichi： taichi代码
- run.py : 编译运行cpp的代码(windows)。
- img.ppm: cpp代码运行后的图片
- taichi.ppm: taichi代码运行后的图片


ppm格式图片推荐使用
https://www.cs.rhodes.edu/welshc/COMP141_F16/ppmReader.html
在线查看


1. ppm格式图片的读写
2. camera和viewport和ray
3. 增加一个球，用求根公式判断相交
4. 根据法线渲染球
5. 增加多个球以及代码重构