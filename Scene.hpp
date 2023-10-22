#pragma once

#include <vector>

#include "Camera.hpp"
#include "Primitives.hpp"
#include "BVH.hpp"

struct LightingAcceleration {
	std::vector<int32_t> prims;
	LightingAcceleration() {}
	LightingAcceleration(const std::vector<Sphere>& src_prims, const std::vector<Material>& material) : prims{} {
		for (int32_t i = 0; i < src_prims.size(); i++) {
			if (auto& em = material[src_prims[i].material_ID].emission; glm::dot(em, em) > 0.0f) prims.push_back(i);
		}
	}
};

struct Scene {
	std::vector<Sphere> geometry;
	std::vector<Material> material;
	LightingAcceleration lighting_acceleration;
	Camera camera;
	Sky sky;
	BoundingVolumeHierarchy<Sphere> acceleration_structure;
};
