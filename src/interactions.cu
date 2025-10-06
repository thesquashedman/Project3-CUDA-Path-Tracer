#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

#include <glm/gtx/color_space.hpp> 

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

//Functions gotten from https://cwyman.org/code/dxrTutors/tutors/Tutor14/tutorial14.md.html
float ggxNormalDistribution(float NdotH, float roughness)
{
    float a2 = roughness * roughness;
    float d = ((NdotH * a2 - NdotH) * NdotH + 1);
    return a2 / (d * d * PI);
}
float schlickMaskingTerm(float NdotL, float NdotV, float roughness)
{
    // Karis notes they use alpha / 2 (or roughness^2 / 2)
    float k = roughness * roughness / 2;

    // Compute G(v) and G(l).  These equations directly from Schlick 1994
    //     (Though note, Schlick's notation is cryptic and confusing.)
    float g_v = NdotV / (NdotV * (1 - k) + k);
    float g_l = NdotL / (NdotL * (1 - k) + k);
    return g_v * g_l;
}
glm::vec3 schlickFresnel(glm::vec3 f0, float lDotH)
{
    return f0 + (glm::vec3(1.0f, 1.0f, 1.0f) - f0) * pow(1.0f - lDotH, 5.0f);
}
glm::vec3 getGGXMicrofacet(thrust::default_random_engine& rng, float roughness, glm::vec3 hitNorm)
{

    // Get our uniform random numbers
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec2 randVal = glm::vec2(u01(rng), u01(rng));

    glm::vec3 reference_vector = glm::vec3(0.0f, 1.0f, 0.0f);
    if(hitNorm == reference_vector)
		reference_vector = glm::vec3(1.0f, 0.0f, 0.0f);
    // Get an orthonormal basis from the normal
    glm::vec3 B = glm::cross(hitNorm, reference_vector);
    glm::vec3 T = glm::cross(B, hitNorm);

    // GGX NDF sampling
    float a2 = roughness * roughness;
    float cosThetaH = glm::sqrt(max(0.0f, (1.0 - randVal.x) / ((a2 - 1.0) * randVal.x + 1)));
    float sinThetaH = glm::sqrt(max(0.0f, 1.0f - cosThetaH * cosThetaH));
    float phiH = randVal.y * PI * 2.0f;

    // Get our GGX NDF sample (i.e., the half vector)
    return T * (sinThetaH * cos(phiH)) +
        B * (sinThetaH * sin(phiH)) +
        hitNorm * cosThetaH;
}


__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 textureColor,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    //if(m.metallic <= 0 && m.roughness <= 0){
    if(true){
        pathSegment.ray.origin = intersect - pathSegment.ray.direction * 0.001f;
        if (m.hasReflective)
        {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        }
        else
        {

            pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);

        }

        
        float lightTerm = glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegment.color *= (m.color * lightTerm) * 0.5f + ((1.0f - intersect.t * 0.02f) * m.color) * 1.0f;
		pathSegment.color *= textureColor;
        
	}
    else
    {
		//Atempted mutlifaceted model, don't think it actually works
        glm::vec3 F0 = glm::mix(glm::vec3(0.04), m.color, m.metallic);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float diffuseProb = u01(rng);
        float lumDiffuse = max(0.01f, glm::luminosity(m.color));
        float lumSpecular = max(0.01f, glm::luminosity(F0));
		float specularProb = lumSpecular / (lumDiffuse + lumSpecular);
        if (diffuseProb < specularProb)
        {
            pathSegment.ray.origin = intersect - pathSegment.ray.direction * 0.001f;
            pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);

            //color.x *= 255;
			//color.y *= 255;
            /*
			pathSegment.color *= (m.color * (1.0f - specularProb)) / (diffuseProb * (1 - specularProb));
			pathSegment.color *= textureColor;
            */

            float lightTerm = glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f));
            pathSegment.color *= (m.color * lightTerm) * 0.5f + ((1.0f - intersect.t * 0.02f) * m.color) * 1.0f;
            pathSegment.color *= textureColor;
        }
        else
        {
            /**/
			glm::vec3 halfVector = getGGXMicrofacet(rng, m.roughness, normal);
            
			glm::vec3 viewDir = -pathSegment.ray.direction;
			glm::vec3 reflectDir = glm::reflect(-viewDir, halfVector);
			pathSegment.ray.origin = intersect + normal * 0.001f;
            pathSegment.ray.direction = reflectDir;
            
			float NdotL = glm::max(glm::dot(normal, pathSegment.ray.direction), 0.0f);
			float NdotV = glm::max(glm::dot(normal, viewDir), 0.0f);
			float NdotH = glm::max(glm::dot(normal, halfVector), 0.0f);
			float LdotH = glm::max(glm::dot(pathSegment.ray.direction, halfVector), 0.0f);

			float D = ggxNormalDistribution(NdotH, m.roughness);
			float G = schlickMaskingTerm(NdotL, NdotV, m.roughness);
			glm::vec3 F = schlickFresnel(F0, LdotH);
			glm::vec3 ggxTerm = (D * G * F) / (4 * NdotL * NdotV);


			float ggxProb = (D * NdotH) / (4 * LdotH);
            pathSegment.color *= (m.color * NdotL * ggxTerm ) / (ggxProb * (1 - diffuseProb));
			pathSegment.color *= textureColor;
            
        }
        
    }
    pathSegment.remainingBounces--;
    


}
