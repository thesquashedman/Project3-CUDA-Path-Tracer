CUDA Path Tracer
================
<img width="800" height="800" alt="pavelscene 2025-10-08_16-02-07z 5000samp" src="https://github.com/user-attachments/assets/147e2cdf-104c-444b-b4f8-e1cd9d0fff00" />

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Pavel Peev
* Tested on: Windows 11, Intel Core Ultra 5 225f, NVIDIA GeForce RTX 5060

### Introduction
This is a CUDA based ray tracer. It supports loading of GLTF meshes and their diffuse textures using tinygltf. It supports two types of materials, a diffuse material and a perfectly specular material. It is optimized with stream compaction to eliminate finished rays, and a BVH spatial structure to speed up triangle ray intersections. Additionally, material sorting can be turned on, which improves performance with larger amounts of emmisive objects.

## GLTF Mesh loading with textures

### Dark Knight
https://sketchfab.com/3d-models/dark-knight-e2208bdc46304f6faa18728778986f35
<img width="600" height="600" alt="cornell 2025-10-08_16-34-02z 5000samp" src="https://github.com/user-attachments/assets/ffc4b110-d776-402d-b15a-8ded2ce6d54e" />

### Rat
https://sketchfab.com/3d-models/rat-dd5d6fbd6edc42e9950778a4ea1fd352
<img width="600" height="600" alt="cornell 2025-10-08_16-46-08z 5000samp" src="https://github.com/user-attachments/assets/80af86c2-37fc-495b-843c-b6fc86c0ee0f" />

## Specular Material
<img width="600" height="600" alt="cornell 2025-09-27_09-20-49z 5000samp" src="https://github.com/user-attachments/assets/7866cb1f-654e-46f4-a407-be92631bf61d" />

<img width="600" height="600" alt="cornell 2025-10-08_17-03-50z 5000samp" src="https://github.com/user-attachments/assets/fbfde5e2-57c9-4514-8194-f385a7297cba" />




## Performance Analysis

### Meshes, BVH, and Mat Sort
Firstly, we compare the render times between a sphere primitive, the dark knight model (44k triangles), and the rat model (100k triangles, with emmisive elements), within an open cornell box. We turn the BVH on and compare the performance between models, as well as the impacts of sorting the materials before shading.

<img width="500" height="500" alt="bargraphmaker net-bargraph" src="https://github.com/user-attachments/assets/315f5418-e199-4546-8023-b19ef0e4a7ec" />

As seen in the graph, the rendering cost for the mesh models is around 5 times for expensive than the sphere primitive. However, between the models, the rendering times the Rat is around 1.5 times more expensive to render than the Dark Knight without material sorting, and a lot closer in costs with material sorting. We see that in a model where most of the elements are the same material (Dark Knight), sorting the materials adds more to the rendering cost, while in a model with more material variation (Rat) we see it improve the rendering time, making it comparable to the dark knight.

To better illustrate the performance gains from BVH, below is a graph showcasing render times with BVH turned off.

<img width="500" height="500" alt="bargraphmaker net-bargraph(1)" src="https://github.com/user-attachments/assets/6d70d73e-9af4-4750-9e67-e822c5293536" />

As seen above, the costs to render a single frame skyrocket. As it turns out, having to test a ray's intersection with hundreds of thousands of triangles is expensive. The BVH helps alleviate that by significantly reducing the number of triangles (logn instead of n) we have to test, at a small memory overhead.

### Stream Compaction

Using stream compaction, we can run the ray intersection kernel on only the rays that are still working (ones that haven't hit a light or the sky), saving gpu resources. In our scenes, we generate 640,000 rays (one for each pixel). Below is how many we rays we eliminate on the first bounce for two different scenes.

|   Closed Scene  |  Open Scene    |
|--------|------------|
| <img width="200" height="200" alt="pavelscene 2025-10-08_15-42-26z 5000samp" src="https://github.com/user-attachments/assets/d56b3977-ad76-4f72-9699-7464ecc9f32e" />| <img width="200" height="200" alt="cornell 2025-10-09_02-06-51z 5000samp" src="https://github.com/user-attachments/assets/6a0abb18-48c1-4e1f-9a71-1d6776755d9b" /> |
| 42883 rays eliminated | 117192 rays eliminated |

As can be seen above, a significant amount of rays are eliminated , around 1/12 for the closed scene and 1/6 for the more open scene. With less rays, the less busy the memory bus is, leading to an improved performance overall.

## Gallery 

<img width="600" height="600" alt="CoverRender" src="https://github.com/user-attachments/assets/116c8b7f-4fa1-4567-a07a-da28802a2830" />

<img width="600" height="600" alt="pavelscene 2025-10-08_15-42-26z 5000samp" src="https://github.com/user-attachments/assets/d56b3977-ad76-4f72-9699-7464ecc9f32e" />

<img width="600" height="600" alt="cornell 2025-10-09_02-06-51z 5000samp" src="https://github.com/user-attachments/assets/6a0abb18-48c1-4e1f-9a71-1d6776755d9b" />

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
