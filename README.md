CUDA Path Tracer
================
<img width="800" height="800" alt="pavelscene 2025-10-08_16-02-07z 5000samp" src="https://github.com/user-attachments/assets/147e2cdf-104c-444b-b4f8-e1cd9d0fff00" />

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Pavel Peev
* Tested on: Windows 11, Intel Core Ultra 5 225f, NVIDIA GeForce RTX 5060

### Introduction
This is a CUDA based ray tracer. It supports loading of GLTF meshes and their diffuse textures using tinygltf. It supports two types of materials, a diffuse material and a perfectly specular material. It is optimized with stream compaction to eliminate finished rays, and a BVH spatial structure to speed up triangle ray intersections. It gives the option to optimize by performing a material sort, but due to only supporting two materials of similiar cost, very little performance gain is achieved.

## GLTF Mesh loading with textures

### Dark Knight
https://sketchfab.com/3d-models/dark-knight-e2208bdc46304f6faa18728778986f35
<img width="600" height="600" alt="cornell 2025-10-08_16-34-02z 5000samp" src="https://github.com/user-attachments/assets/ffc4b110-d776-402d-b15a-8ded2ce6d54e" />

### Rat
https://sketchfab.com/3d-models/rat-dd5d6fbd6edc42e9950778a4ea1fd352
<img width="600" height="600" alt="cornell 2025-10-08_16-46-08z 5000samp" src="https://github.com/user-attachments/assets/80af86c2-37fc-495b-843c-b6fc86c0ee0f" />

### Collector
https://sketchfab.com/3d-models/collector-827d2795ae2a4e58bc1313f0d4a3ee48

## Specular Material
<img width="600" height="600" alt="cornell 2025-09-27_09-20-49z 5000samp" src="https://github.com/user-attachments/assets/7866cb1f-654e-46f4-a407-be92631bf61d" />

<img width="600" height="600" alt="cornell 2025-10-08_17-03-50z 5000samp" src="https://github.com/user-attachments/assets/fbfde5e2-57c9-4514-8194-f385a7297cba" />


## Gallery 

<img width="600" height="600" alt="CoverRender" src="https://github.com/user-attachments/assets/116c8b7f-4fa1-4567-a07a-da28802a2830" />

<img width="600" height="600" alt="pavelscene 2025-10-08_15-42-26z 5000samp" src="https://github.com/user-attachments/assets/d56b3977-ad76-4f72-9699-7464ecc9f32e" />

<img width="600" height="600" alt="cornell 2025-10-09_02-06-51z 5000samp" src="https://github.com/user-attachments/assets/6a0abb18-48c1-4e1f-9a71-1d6776755d9b" />


## Performance Analysis

## Models used

### Dark Knight
https://sketchfab.com/3d-models/dark-knight-e2208bdc46304f6faa18728778986f35
### Rat
https://sketchfab.com/3d-models/rat-dd5d6fbd6edc42e9950778a4ea1fd352
### Collector
https://sketchfab.com/3d-models/collector-827d2795ae2a4e58bc1313f0d4a3ee48
### Mario
https://sketchfab.com/3d-models/mario-obj-c549d24b60f74d8f85c7a5cbd2f55d0f
### Low Poly Abandoned Brick Room
https://sketchfab.com/3d-models/low-poly-abandoned-brick-room-dd2a99fd9f3f456cac3680dc7127ac22
### Railway Signal Box - Bytom, Poland
https://sketchfab.com/3d-models/railway-signal-box-bytom-poland-bae9c6f783ac4017a979cb9e3259a12f
### Low Poly Dirt Ground
https://sketchfab.com/3d-models/low-poly-dirt-ground-88bd58f71a4f43688b61c42bdd8934c3
