#pragma once

#include "Core.hpp"

struct Projection {
	Projection(uint32_t W, uint32_t H, float focal_length, float focus_distance, float f_number) 
		: focal_length(focal_length), focus_distance(focus_distance), f_number(f_number) {
		Resize(W, H);
		UpdateLens();
	}

	//field of view in radians
	[[msvc::forceinline]] static float calc_field_of_view(float focal_length, float sensor_size = 24.0f) {
		return 2.0f * static_cast<float>(glm::atan((sensor_size / 2.0f) / focal_length));
	}

	[[msvc::forceinline]] static float calc_aperture(float focal_length, float f_number) {
		return focal_length / (2.0f * f_number);
	}

	[[msvc::forceinline]] void UpdateLens() {
		static constexpr float sensor_size = 24.0f/*mm*/;
		inv_half_tan = (-2.0f / sensor_size) * focal_length; //-1/(tan(0.5 * field_of_view_radians))
		aperture_radius = calc_aperture(focal_length, f_number);
		z = half_height * inv_half_tan;
	}

	[[msvc::forceinline]] void Resize(uint32_t W, uint32_t H) {
		half_height = static_cast<float>(H) * 0.5f;
		half_width = static_cast<float>(W) * 0.5f;
		z = half_height * inv_half_tan;
	}
	float half_height;
	float half_width;
	float z;
	float aperture_radius;
	float inv_half_tan;

	float focal_length;
	float focus_distance;
	float f_number;

	float near = 0.0f;
	float far = 1000.0f;
};

struct View {
	View(glm::vec3 eye, glm::vec3 forward) : pos(eye) {
		orient = glm::quatLookAt(glm::normalize(forward), glm::vec3{0.0f, 1.0f, 0.0f});
	}
	[[msvc::forceinline]] void Rotate(glm::vec3 /*pitch, yaw, roll*/ angles) {
		orient = glm::conjugate(glm::normalize(glm::quat{angles} * glm::conjugate(orient)));
	}
	[[msvc::forceinline]] void Translate(glm::vec3 local_transform) {
		pos += orient * local_transform;
	}
	glm::quat orient;
	glm::vec3 pos;
};

struct Camera {
	Camera(
		glm::vec3 eye = {0, 0, 0}, glm::vec3 direction = {0, 0, -1}, uint32_t width = 1, uint32_t height = 1,
		float focal_length = 50.0f, float focus_distance = 1.0f, float f_number = 16.0f, float exposure = 1.0f) :
		view(eye, direction), projection(width, height, focal_length, focus_distance, f_number), exp(exposure) {}
	void Resize(uint32_t width, uint32_t height) {
		projection.Resize(width, height);
	}
	void RotateLocal(glm::vec3 /*pitch, yaw, roll*/  angles) {
    	view.Rotate(angles);
	}
	void TranslateLocal(glm::vec3 local_transform) {
		view.Translate(local_transform);
	}
	View view;
	Projection projection;
	float exp;


	struct { glm::vec3 origin, dir; } generate_ray(int32_t x, int32_t y, const float* const __restrict samples) const noexcept {
		return { view.pos, 
			glm::normalize(glm::rotate(view.orient, glm::vec3{
				static_cast<float>(x) + samples[0] - projection.half_width,
				static_cast<float>(y) + samples[1] - projection.half_height,
				projection.z
			}))
		};
	}
};