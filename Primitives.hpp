#pragma once

#include <format>

#include "Core.hpp"

struct alignas(16) Sphere {
	glm::vec3 position;
	float radius_sq;
	int32_t material_ID;

	struct bounds_ret{ glm::vec3 min, max; };
	bounds_ret bounds() const noexcept {
		float r = sqrt(radius_sq);
		return {position - r, position + r};
	}
};
struct alignas(32) Material {
	glm::vec3 albedo;
	glm::vec3 F0;
	glm::vec3 F80;
	glm::vec3 emission;
	glm::vec3 transmission;

	float roughness;
	float IOR_minus_one;
};

struct Sky {
	glm::vec3 ambient_color;
	int32_t hdri_width = 0, hdri_height = 0, hdri_channels = 0;
	float* hdri_data = nullptr;
	float hdri_fwidth, hdri_fheight;

	glm::vec3 operator()(float x, float y, float z) const noexcept {
		glm::vec2 equirect_coords {
			hdri_fwidth  * (0.5f + glm::one_over_two_pi<float>() * fast_atan2(z, x)),
			hdri_fheight * (0.5f - glm::one_over_pi<float>()     * fast_asin (y))
		};
		float* sky_value = hdri_data + 4 * (static_cast<int32_t>(equirect_coords.y) * hdri_width + static_cast<int32_t>(equirect_coords.x));
		return {
			sky_value[0] * ambient_color.r,
			sky_value[1] * ambient_color.g,
			sky_value[2] * ambient_color.b
		};
	}
};

struct AABB {
	alignas(16) glm::vec3 min_bound;
	alignas(16) glm::vec3 max_bound;

	AABB() : min_bound(0.0f), max_bound(0.0f) {}
	AABB(glm::vec3 min, glm::vec3 max) : min_bound(min), max_bound(max) {}
	AABB(Sphere sphere) : min_bound(sphere.position - sqrt(sphere.radius_sq)), max_bound(sphere.position + sqrt(sphere.radius_sq)){}
	glm::vec3 size() const { return abs(max_bound - min_bound); }
	float area() const { auto s = size(); return 2.0f * dot(s, glm::vec3{s.y,  s.z, s.x}); }
	float volume() const { auto s = size(); return s.x * s.y * s.z; }
	uint32_t largest_axis() const {
		auto s = size();
		uint32_t axis = 0; float max_size = 0.0f;
		for (uint32_t i = 0; i < 3; i++){
			if (s[i] > max_size) {
				max_size = s[i];
				axis = i;
			}
		}
		return axis;
	}
	friend AABB operator+(AABB a, AABB b) {
		return { min(a.min_bound, b.min_bound), max(a.max_bound, b.max_bound) };
	}
	friend std::ostream& operator<<(std::ostream& os, const AABB& obj) {
		return os << std::format("[({}:{}),({}:{}),({}:{})]",
			obj.min_bound.x, obj.max_bound.x,
			obj.min_bound.y, obj.max_bound.y,
			obj.min_bound.z, obj.max_bound.z
		);
	}
};

struct BS {
	alignas(16) glm::vec3 center;
	float radius;

	BS() : center(0.0f), radius(0.0f) {}
	BS(glm::vec3 p, float r) : center(p), radius(r) {}
	BS(Sphere sphere) : center(sphere.position), radius(sqrt(sphere.radius_sq) + 0.00001f) {}
	glm::vec3 size() const { return glm::vec3{radius}; }
	float area() const { return static_cast<float>(4.0 * glm::pi<double>()) * radius * radius; }
	float volume() const { return static_cast<float>(4.0/3.0 * glm::pi<double>()) * radius * radius * radius; }
	uint32_t largest_axis() const { return 1; }
	friend BS operator+(BS a, BS b) {
		if (a.radius == 0.0f) return b;
		if(b.radius == 0.0f) return a;
		float dist = distance(a.center, b.center);
		if (dist < 0.0001f) return (a.radius > b.radius ? a : b);
		if (dist + a.radius < b.radius) return b;
		if (dist + b.radius < a.radius) return a;
		float combined_radius = (a.radius + b.radius + dist ) / 2.0f + 0.00001f;
		glm::vec3 combined_center = a.center + (b.center - a.center) * (combined_radius - a.radius) / dist;
		return {combined_center, combined_radius};
	}
	friend std::ostream& operator<<(std::ostream& os, const BS& obj){ 
		return os << std::format("[{},{},{}//R:{}]", obj.center.x, obj.center.y, obj.center.z, obj.radius); 
	}
};
