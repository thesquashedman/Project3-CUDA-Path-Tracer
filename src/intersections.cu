#include "intersections.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    /*
    if (!outside)
    {
        normal = -normal;
    }
    */
    return glm::length(r.origin - intersectionPoint);
}
//https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html
__host__ __device__ float triangleIntersectionTest(Tri triangle, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside)
{
    //TODO compute outside

    glm::vec3 N = glm::cross(triangle.pos[1] - triangle.pos[0], triangle.pos[2] - triangle.pos[0]);
    // Step 1: Finding P

    // Check if the ray and plane are parallel


    float NDotRayDirection = glm::dot(N, r.direction);

    if (glm::abs(NDotRayDirection) < 0.00001) // Almost 0
        return -1; // They are parallel, so they don't intersect!

    // Compute d parameter using equation 2
    float d = -glm::dot(N, triangle.pos[0]);
    // Compute t (equation 3)
    float t = (-glm::dot(N, r.origin) + d) / NDotRayDirection;

    // Check if the triangle is behind the ray
    if (t < 0) return -1; // The triangle is behind

    // Compute the intersection point using equation 1
    glm::vec3 P = r.origin + t * r.direction;
    // Step 2: Inside-Outside Test
    glm::vec3 Ne;
    // Test sidedness of P w.r.t. edge v0v1
    glm::vec3 v0p = P - triangle.pos[0];
    Ne = glm::cross(triangle.pos[1] - triangle.pos[0], v0p);
    if (glm::dot(N, Ne) < 0) return -1;  // P is on the right side
    // Test sidedness of P w.r.t. edge v2v1
    glm::vec3 v1p = P - triangle.pos[1];
    Ne = glm::cross(triangle.pos[2] - triangle.pos[1], v1p);
    if (glm::dot(N, Ne) < 0) return -1;  // P is on the right side

    // Test sidedness of P w.r.t. edge v2v0
    glm::vec3 v2p = P - triangle.pos[2];
    Ne = glm::cross(triangle.pos[0] - triangle.pos[2], v2p);
    if (glm::dot(N, Ne) < 0) return -1;  // P is on the right side

    intersectionPoint = P;
    normal = glm::normalize(N);

    return t;
}
__host__ __device__ float bvhIntersectionTest(
    BVHNode* node,
    int initialIndex,
    Ray r,
    Tri* tris,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
	glm::vec2& uv,
    int& triangleIndex,
    bool& outside)
{
	int stack[64];
	stack[0] = initialIndex;
    int curr = 0;
	float closestT = FLT_MAX;
	bool hit = false;

	
    while (curr >=0)
    {

        int index = stack[curr];
        curr--;


        
        glm::vec3 invDir = 1.0f / r.direction;
        glm::vec3 tmin = (node[index].bboxMin - r.origin) * invDir;
        glm::vec3 tmax = (node[index].bboxMax - r.origin) * invDir;

        float tminx = glm::min(tmin.x, tmax.x);
        float tmaxx = glm::max(tmin.x, tmax.x);

        float tminy = glm::min(tmin.y, tmax.y);
        float tmaxy = glm::max(tmin.y, tmax.y);

        float tminz = glm::min(tmin.z, tmax.z);
        float tmaxz = glm::max(tmin.z, tmax.z);
        float t0 = glm::max(tminx, glm::max(tminy, tminz));
        float t1 = glm::min(tmaxx, glm::min(tmaxy, tmaxz));
        

        if (t0 < t1)
        {
            //If leaf node, intersect with triangles
            if (node[index].left == -1 && node[index].right == -1)
            {

                for (int i = node[index].triIndexStart; i <= node[index].triIndexEnd; i++)
                {
                    float t = -1;
					glm::vec3 tmp_intersect;
					glm::vec3 tmp_normal;
					glm::vec2 tmp_uv;


                    glm::vec3 baryPosition;
                    bool intersects = glm::intersectRayTriangle(r.origin, r.direction,
                        tris[i].pos[0], tris[i].pos[1], tris[i].pos[2], baryPosition);

                    if (intersects)
                    {

                        t = baryPosition.z;
                        tmp_intersect = r.origin + t * r.direction;

                        tmp_normal = glm::normalize(glm::cross(tris[i].pos[1] - tris[i].pos[0], tris[i].pos[2] - tris[i].pos[0]));
                        glm::vec2 uv0 = tris[i].uv[0];
						glm::vec2 uv1 = tris[i].uv[1];
						glm::vec2 uv2 = tris[i].uv[2];
						tmp_uv = baryPosition.x * tris[i].uv[1] + baryPosition.y * tris[i].uv[2] + (1 - baryPosition.x - baryPosition.y) * tris[i].uv[0];


                    }
                    else
                    {
                        t = -1;
                    }


                    if (t > 0.0f && closestT > t)
                    {
                        closestT = t;

                        intersectionPoint = tmp_intersect;
                        normal = tmp_normal;
						uv = tmp_uv;
						triangleIndex = i;
						hit = true;

                    }
                }

            }
            else
            {
				//Else push children to stack. Add right first so left nodes are handles first, for better coherency

                curr++;
				stack[curr] = node[index].right;;
				curr++;
				stack[curr] = node[index].left;


            }


        }

        
	}
	
    if (hit == false)
    {
        return -1;
    }
    return closestT;

}

    



