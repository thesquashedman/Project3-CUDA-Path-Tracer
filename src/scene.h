#pragma once

#include "sceneStructs.h"
#include <vector>
#include <unordered_map>

#include "stb.cpp"


#include "tiny_gltf.h"


class Scene
{
private:
    void loadFromJSON(const std::string& jsonName, const std::string& meshFile);
    void loadGLTF(const std::string& meshFile, glm::mat4 transform);
    void strideOverIndex(const uint8_t* data, int byteOffset, int size, std::vector<glm::vec3>& fillVector);
    void calculateGlobalTransform(tinygltf::Model& model, int nodeId, glm::mat4 parentTransform, std::unordered_map<int, glm::mat4>& nodeToGlobalTransform);
    int createBVH(int triStart, int triEnd, int depth);
public:
    Scene(std::string sceneFile, std::string meshFile);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Tri> tris;
	std::vector<BVHNode> bvhNodes;
	std::vector<Texture> textures;
    std::vector<glm::vec3> textureData;
    RenderState state;

};
