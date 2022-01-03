# TaichiPlayground

用[太极](https://github.com/taichi-dev/taichi)做一些好玩的动画

* [3body.py](3body.py) 基于官方 [nbody](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/nbody.py) 样例修改，模拟三体运动。尝试了官方提供的 taichi.VideoManager 接口生成视频，效率很低；最后用 OBS (Open Broadcaster Software) 直接录制并合成音乐，带了一些风扇杂音: [Youtube录屏](https://youtu.be/sYBcrnLAdpU)
* [elastic_rope.py](elastic_rope.py) 弹力绳固定一端掉落动画模拟，做法是把绳子想象成用弹簧串联的N个质点。B站录屏：[200质点有风阻](https://www.bilibili.com/video/BV1GZ4y1S7BV)，[100质点无风阻](https://www.bilibili.com/video/BV17r4y1U7h7?spm_id_from=333.999.0.0)
