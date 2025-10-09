#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

using namespace std;
using json = nlohmann::json;

Scene::Scene(string sceneFile, string meshFile)
{
    cout << "Reading scene from " << sceneFile << " ..." << endl;
    cout << " " << endl;
    auto ext = sceneFile.substr(sceneFile.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(sceneFile, meshFile);
        return;
    }
    else
    {
        cout << "Couldn't read from " << sceneFile << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName, const std::string& meshFile)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = true;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh")
        {
            newGeom.type = MESH;
            const auto& meshName = p["NAME"];
			string meshFilePath = meshFile + "/" +  meshName.get<string>();
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
			glm::mat4 transform = utilityCore::buildTransformationMatrix(glm::vec3(trans[0], trans[1], trans[2]), glm::vec3(rotat[0], rotat[1], rotat[2]), glm::vec3(scale[0], scale[1], scale[2]));
            /*
            newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            */
			loadGLTF(meshFilePath, transform);

        }
        if (type != "mesh")
        {
            newGeom.materialid = MatNameToID[p["MATERIAL"]];

            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            geoms.push_back(newGeom);
        }
    }
    //Create BVH
#ifdef BVH_SOLUTION
    createBVH(0, tris.size(), 0);
#endif

   

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
void Scene::loadGLTF(const std::string& meshFile, glm::mat4 transform)
{
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;
    std::unordered_map<int, glm::mat4> nodeToGlobalTransform;


    loader.LoadASCIIFromFile(&model, &err, &warn, meshFile);
    glm::mat4 globalTransform = transform;
	int rootNode = model.scenes[model.defaultScene].nodes[0];
	calculateGlobalTransform(model, rootNode, globalTransform, nodeToGlobalTransform);
	//Load the textures into texturdeData
    for(int i = 0; i < model.textures.size(); i++)
    {
        const tinygltf::Texture& tex = model.textures[i];
        const tinygltf::Image& img = model.images[tex.source];
        Texture newTexture;
        newTexture.width = img.width;
        newTexture.height = img.height;
        //newTexture.component = img.component;
        //newTexture.bits = img.bits;
        //newTexture.pixel_type = img.pixel_type;
        //newTexture.imageData = img.image;

        //Copy image data to textureData
        int oldSize = textureData.size();
        textureData.resize(oldSize + (img.width * img.height));
        for(int j = 0; j < img.width * img.height; j++)
        {
            if(img.component == 3)
            {
                textureData[oldSize + j] = glm::vec3(img.image[3 * j] / 255.0f, img.image[3 * j + 1] / 255.0f, img.image[3 * j + 2] / 255.0f);
            }
            else if(img.component == 4)
            {
                textureData[oldSize + j] = glm::vec3(img.image[4 * j] / 255.0f, img.image[4 * j + 1] / 255.0f, img.image[4 * j + 2] / 255.0f);
            }


          
        }
		newTexture.dataStart = oldSize;
		textures.push_back(newTexture);
	}
       
    

	//For each node with a mesh, create a geom
	//Create Geom for each mesh
    for(int i = 0; i < model.nodes.size(); i++)
    {
        tinygltf::Node& node = model.nodes[i];
	

        if(node.mesh < 0) continue;


        const tinygltf::Mesh& mesh = model.meshes[node.mesh];


        //Create a new geom for each mesh
		Geom newGeom;
        newGeom.type = MESH;

        
		newGeom.transform = nodeToGlobalTransform[i];
		newGeom.inverseTransform = glm::inverse(newGeom.transform);
		newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        //For creating bounding box
		glm::vec3 pMin = glm::vec3(FLT_MAX);
		glm::vec3 pMax = glm::vec3(-FLT_MAX);

        newGeom.triIndexStart = tris.size();

        int indicesByteStride = 0;
        int indicesCount = 0;
		const uint8_t* indicesDataAddress = 0;

        //Load materials



        


		



        //Go through each primtive in the mesh (each of the points)
        std::vector<glm::vec3> positions;
        std::vector<glm::vec2> uvs;
        for(int j = 0; j < mesh.primitives.size(); j += 3)
        {
            const tinygltf::Primitive& primitive = mesh.primitives[j];

            if (primitive.material >= 0 && primitive.material < model.materials.size()) {
                Material newMaterial{};
                const tinygltf::Material& mat = model.materials[primitive.material];
                if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4)
                {
                    newMaterial.color = glm::vec3(mat.pbrMetallicRoughness.baseColorFactor[0],
                        mat.pbrMetallicRoughness.baseColorFactor[1],
                        mat.pbrMetallicRoughness.baseColorFactor[2]);

                    newMaterial.metallic = mat.pbrMetallicRoughness.metallicFactor;
                    newMaterial.roughness = mat.pbrMetallicRoughness.roughnessFactor;


                }
                if (mat.pbrMetallicRoughness.baseColorTexture.index >= 0)
                {
                    newMaterial.textureId = mat.pbrMetallicRoughness.baseColorTexture.index;
                }
                //emmision
                
                if(mat.emissiveFactor.size() == 3)
                {
                    if(mat.emissiveFactor[0] > 0 || mat.emissiveFactor[1] > 0 || mat.emissiveFactor[2] > 0)
                    {
                        newMaterial.emittance = 1.0f;
                        newMaterial.color = glm::vec3(mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]);
					}
				}
                materials.push_back(newMaterial);
            }





            

            const auto& indicesAccessor = model.accessors[primitive.indices];
            const auto& indicesBufferView = model.bufferViews[indicesAccessor.bufferView];
            const auto& indicesBuffer = model.buffers[indicesBufferView.buffer];
            indicesDataAddress = indicesBuffer.data.data() + indicesBufferView.byteOffset + indicesAccessor.byteOffset;
            indicesByteStride = indicesAccessor.ByteStride(indicesBufferView);
            indicesCount = indicesAccessor.count;
           
            for (auto& attrib : primitive.attributes)
            {
                
                const auto attribAccessor = model.accessors[attrib.second];
                const auto& bufferView = model.bufferViews[attribAccessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];
                const auto dataPtr = buffer.data.data() + bufferView.byteOffset + attribAccessor.byteOffset;
                const auto byte_stride = attribAccessor.ByteStride(bufferView);
                const auto count = attribAccessor.count;

                if (attrib.first == "POSITION")
                {
                    pMin.x = attribAccessor.minValues[0];
                    pMin.y = attribAccessor.minValues[1];
                    pMin.z = attribAccessor.minValues[2];
                    pMax.x = attribAccessor.maxValues[0];
                    pMax.y = attribAccessor.maxValues[1];
                    pMax.z = attribAccessor.maxValues[2];

					strideOverIndex(dataPtr, byte_stride, count, positions);
					

                    //

                }
                
                else if (attrib.first == "TEXCOORD_0")
                {
                    
                    
                    for (int i = 0; i < count; i++)
                    {
                        const float* buf = reinterpret_cast<const float*>(dataPtr + i * byte_stride);
						float x = buf[0];
						float y = buf[1];

						x = x - floor(x);
						y = y - floor(y);
						//y = 1.0f - y; //Flip the y coordinate
                        glm::vec2 vec = glm::vec2(x, y);
                        if(vec.x > 1.0f || vec.y > 1.0f)
                        {
                            std::cout << "Warning: UV coordinates greater than 1.0f found. This may cause issues with texture mapping." << std::endl;
						}

                        uvs.push_back(vec);
                    }
                    
				}
                
                
			}
            
        }
        for (int k = 0; k < indicesCount; k += 3)
        {
            Tri newTri;
			int index1 = reinterpret_cast<const uint16_t*>(indicesDataAddress + k * indicesByteStride)[0];
			int index2 = reinterpret_cast<const uint16_t*>(indicesDataAddress + (k + 1) * indicesByteStride)[0];
			int index3 = reinterpret_cast<const uint16_t*>(indicesDataAddress + (k + 2) * indicesByteStride)[0];

            newTri.pos[0] = glm::vec3(newGeom.transform * glm::vec4(positions[index1], 1));
			newTri.pos[1] = glm::vec3(newGeom.transform * glm::vec4(positions[index2], 1));
			newTri.pos[2] = glm::vec3(newGeom.transform * glm::vec4(positions[index3], 1));
            if(uvs.size() > 0)
            {
                newTri.uv[0] = uvs[index1];
                newTri.uv[1] = uvs[index2];
				newTri.uv[2] = uvs[index3];
            }

			newTri.geomId = geoms.size();
            tris.push_back(newTri);
        }
        newGeom.triIndexEnd = tris.size();
		newGeom.materialid = materials.size() - 1; //Last material added
		std::cout << "Loaded mesh with " << (newGeom.triIndexEnd - newGeom.triIndexStart) << " triangles." << std::endl;
		geoms.push_back(newGeom);
	}

}
void Scene::strideOverIndex(const uint8_t* data, int byteOffset, int size, std::vector<glm::vec3>& fillVector)
{
    for(int i = 0; i < size; i++)
    {
        const float* buf = reinterpret_cast<const float*>(data + i * byteOffset);
        glm::vec3 vec = glm::vec3(buf[0], buf[1], buf[2]);
        fillVector.push_back(vec);
	}
}
void Scene::calculateGlobalTransform(tinygltf::Model& model,int nodeId, glm::mat4 parentTransform, std::unordered_map<int, glm::mat4>& nodeToGlobalTransform)
{
    glm::mat4 localTransform = glm::mat4(1.0f);

	tinygltf::Node& node = model.nodes[nodeId];

    if (node.translation.size() == 3)
        localTransform = glm::translate(localTransform, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
    if (node.rotation.size() == 4)
    {
        glm::quat q = glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
        localTransform *= glm::mat4_cast(q);
    }
    if (node.scale.size() == 3)
        localTransform = glm::scale(localTransform, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
    
    glm::mat4 globalTransform = parentTransform * localTransform;
    
    nodeToGlobalTransform[nodeId] = globalTransform;
    for (int i = 0; i < node.children.size(); i++)
    {
        int childNode = node.children[i];
		calculateGlobalTransform(model, childNode, globalTransform, nodeToGlobalTransform);
    }
}
int Scene::createBVH(int triStart, int triEnd, int depth)
{
    
	BVHNode node{};
	int currentIndex = bvhNodes.size();
	bvhNodes.push_back(node);
	node.bboxMin = glm::vec3(FLT_MAX);
	node.bboxMax = glm::vec3(-FLT_MAX);
    node.triIndexStart = triStart;
    node.triIndexEnd = triEnd;
    int maxTrisPerLeaf = 10;
	//Calculate bounding box
    for (int i = triStart; i < triEnd; i++)
    {
        Tri& tri = tris[i];
        for (int j = 0; j < 3; j++)
        {
            node.bboxMin = glm::min(node.bboxMin, tri.pos[j]);
            node.bboxMax = glm::max(node.bboxMax, tri.pos[j]);
        }
	}
    //Leaf node
    if(triEnd - triStart <= maxTrisPerLeaf)
    {

		node.left = -1;
		node.right = -1;
        
	}
    else 
    {
        /*
		glm::vec3 extent = node.bboxMax - node.bboxMin;
		int axis = 0;
		if (extent.y > extent.x && extent.y > extent.z)
			axis = 1;
		else if (extent.z > extent.x && extent.z > extent.y)
			axis = 2;
	    
        vector<Tri> tempTrisLeft;
		vector<Tri> tempTrisRight;
        for (int i = triStart; i < triEnd; i++)
        {
			//Calculate centroid
			glm::vec3 centroid = (tris[i].pos[0] + tris[i].pos[1] + tris[i].pos[2]) / 3.0f;
            if (centroid[axis] < (node.bboxMin[axis] + node.bboxMax[axis]) / 2.0f)
            {
                tempTrisLeft.push_back(tris[i]);
            }
            else
            {
                tempTrisRight.push_back(tris[i]);
			}

        }
		//replace tris in main array
		int leftStart = triStart;
		int leftEnd = leftStart + tempTrisLeft.size();
		int rightStart = leftEnd;
		int rightEnd = rightStart + tempTrisRight.size();
		for (int i = leftStart; i < tempTrisLeft.size(); i++)
			tris[i] = tempTrisLeft[i - leftStart];
		for (int i = rightStart; i < rightEnd; i++)
			tris[i] = tempTrisRight[i - rightStart];
           */
        glm::vec3 extent = node.bboxMax - node.bboxMin;
        int axis = 0;
        if (extent.y > extent.x && extent.y > extent.z)
            axis = 1;
        else if (extent.z > extent.x && extent.z > extent.y)
            axis = 2;


        auto cmpFunc = [axis](const Tri& a, const Tri& b) {
            glm::vec3 center_a = (a.pos[0] + a.pos[1] + a.pos[2]) / 3.0f;
            glm::vec3 center_b = (b.pos[0] + b.pos[1] + b.pos[2]) / 3.0f;
            return center_a[axis] < center_b[axis];
            };

        std::sort(tris.begin() + triStart, tris.begin() + triEnd, cmpFunc);

        int mid = (triStart + triEnd) >> 1;

        node.left = createBVH(triStart, mid, depth + 1);
	    node.right = createBVH(mid, triEnd, depth + 1);
        





    }

    bvhNodes[currentIndex] = node;

	return currentIndex;
}
