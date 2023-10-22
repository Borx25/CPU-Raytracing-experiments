#pragma once

#include "Core.hpp"
#include "Random.hpp"
#include <variant>
#include <span>

template<typename T>
T median(T a, T b, T c) {
	using std::max; using std::min;
	return max(min(a, b), min(max(a, b), c));
}
template<typename T>
T median(T a, T b, T c, T d, T e) {
	using std::max; using std::min;
	return median(
		max(min(a, b), min(c, d)),
		min(max(a, b), max(c, d)),
		e
	);
}
/*----------------------------------
-- RESAMPLED IMPORTANCE SAMPLING --
-----------------------------------*/
struct alignas(8) RIS_Sample { int32_t sample = -1; float weight = 0.0f; };

struct alignas(16) Reservoir {
	RIS_Sample selected{};
	float weight_sum = 0.0f;
	int32_t count = 0; //number of samples seen so far
	inline void Update(RIS_Sample x, float random_01, int32_t num_samples = 1) noexcept {
		count += num_samples;
		weight_sum += x.weight;
		if (random_01 < (x.weight / weight_sum))
			selected.sample = x.sample;
	}
};

//En el pdf hay varias notas sobre modificaciones para que sea unbiased si hago spatial reuse (supongo q en mi setup seria dentro de un tile
//https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf
//src dist returns any tuple of int32_t for the sample and its reciprocal probability { Xi, 1/p(Xi) }
template<size_t count> static inline RIS_Sample RIS(auto src_dist, auto weight_sample, uint32_t* rng_state) noexcept {
	Reservoir r{};
	for (uint32_t i = 0; i < count; i++)
	{
		auto[Xi, rpXi] = src_dist(i);
		r.Update(RIS_Sample{Xi,  weight_sample(Xi) * rpXi}, rand_unit_float(rng_state));
	}
	if(r.selected.sample < 0) return {-1, 0.0f};
	const float w = weight_sample(r.selected.sample);
	if (w <= 0.0f) return {-1, 0.0f};
	r.selected.weight = r.weight_sum / (static_cast<float>(r.count) * w);
	return r.selected;
};

Reservoir CombineReservoirs(const std::span<Reservoir> reservoirs, auto weight_sample, uint32_t* rng_state) noexcept {
	Reservoir r = reservoirs[0];
	for (uint32_t i = 1; i < reservoirs.size(); i++)
	{
		r.Update(RIS_Sample{
				reservoirs[i].selected.sample,
				weight_sample(reservoirs[i].selected.sample) * reservoirs[i].selected.weight * static_cast<float>(reservoirs[i].count) 
			}, 
			rand_unit_float(rng_state),
			reservoirs[i].count
		);
	}
	if (r.selected.sample < 0) return {-1, 0.0f};
	const float w = weight_sample(r.selected.sample);
	if (w <= 0.0f) return {-1, 0.0f};
	r.selected.weight = r.weight_sum / (static_cast<float>(r.count) * w);
	return r;
};
/*---------------------------------
----------- MAPPINGS ------------
---------------------------------*/
static inline glm::vec3 spherical_to_cartesian(float phi_over_2pi, float sin_theta, float cos_theta) noexcept {
	float cos_phi, sin_phi; fast_sincos(phi_over_2pi * glm::two_pi<float>(), &sin_phi, &cos_phi);
	return {
		sin_theta * cos_phi,
		sin_theta * sin_phi,
		cos_theta
	};
}
static inline glm::vec2 polar_to_cartesian(float phi_over_2pi, float rho) noexcept {
	float cos_phi, sin_phi; fast_sincos(phi_over_2pi * glm::two_pi<float>(), &sin_phi, &cos_phi);
	return {
		rho * cos_phi,
		rho * sin_phi
	};
}
glm::vec3 hemisphere(float t, float s) noexcept { //Oriented +Z, cosine weighted
	return spherical_to_cartesian(s, sqrt(t), sqrt(std::max(0.0f, 1.0f - t)));
}
glm::vec3 uniform_hemisphere(float t, float s) noexcept { //Oriented +Z
	return spherical_to_cartesian(s, sqrt(std::max(0.0f, 1.0f - t * t)), t);
}
glm::vec3 uniform_sphere(float t, float s) noexcept {
	t = 1.0f - 2.0f * t;
	return spherical_to_cartesian(s, sqrt(std::max(0.0f, 1.0f - t * t)), t);
}
glm::vec2 disk(float t, float s) noexcept {
	return polar_to_cartesian(s, sqrt(t));
}
/*---------------------------------
--------- TANGENT SPACE ----------
---------------------------------*/
//http://lolengine.net/blog/2013/09/21/picking-orthogonal-vector-combing-coconuts
/* Always works if the input is non-zero.
 * Doesnt require the input to be normalised.
 * Doesnt normalise the output. */
static inline glm::vec3 orthogonal(glm::vec3 v) noexcept {
	return abs(v.x) > abs(v.z) ? glm::vec3(-v.y, v.x, 0.0f) : glm::vec3(0.0f, -v.z, v.y);
}
//https://graphics.pixar.com/library/OrthonormalB/paper.pdf
static inline void orthonormal_basis(const glm::vec3& n, glm::vec3* v2, glm::vec3* v3) {
#if true //dicen que esta branchless es mas rapida pq a diferencia de un metodo que haga branch solo para singularity que seria casi siempre falso aqui es 50% luego branch predictor mal
	//const float s = fast_sign(n.z);// n.z >= 0.0f ? 1.0f : -1.0f;
	//const float a = -1.0f / (s + n.z);
	//const float b = n.x * n.y * a;
	//*v2 = glm::vec3{ 1.0f + s * n.x * n.x * a, s * b , -s * n.x };
	//*v3 = glm::vec3{ b, s + a * n.y * n.y, -n.y };
	float sign = f_and(-0.0f, n.z);
	float s = f_xor(1.0f, sign);
	float z = -1.0f / (s + n.z);
	float s_nx = f_xor(sign, n.x);
	float ny_z = n.y * z;
	float t = n.x * ny_z;
	*v2 = glm::vec3{1.0f + (s_nx * n.x) * z, f_xor(sign, t), -s_nx};
	*v3 = glm::vec3{t, s + ny_z * n.y, -n.y};
#else
	if (n.z < 0.) {
		const float a = -1.0f / (-1.0f + n.z);
		const float b = n.x * n.y * a;
		*v2 = glm::vec3{ 1.0f - n.x * n.x * a, -b, n.x };
		*v3 = glm::vec3{ b, -1.0f + n.y * n.y * a, -n.y };
	} else {
		const float a = -1.0f / (1.0f + n.z);
		const float b = n.x * n.y * a;
		*v2 = glm::vec3{ 1.0f + n.x * n.x * a, b, -n.x };
		*v3 = glm::vec3{ b, 1.0f + n.y * n.y * a, -n.y };
	}
#endif
}


//Calculate quaternion encoding the orthonormal basis (rotates +Z{0,0,1} to N)
//SOURCE: https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
//quaternion Z can be assumed to be 0.0
static inline glm::quat tangent_space(glm::vec3 N) {
	//return glm::rotation(glm::normalize(N), {0.0,0.0,1.0});
	if (N.z < -1.0f + FLT_EPSILON) [[unlikely]] {
		return {0.0, 0.0, 1.0, 0.0};
	} else {
		float s = sqrt(2.0f * (N.z + 1.0f));
		float invs = 1.0f / s;
		return {s * 0.5f, -N.y * invs, N.x * invs, 0.0f};
	}
}
//rotate by conjugate of T given T.z == 0.0 and T is normalized
static inline glm::vec3 to_local(glm::quat T, glm::vec3 v) {
	//return glm::rotate(glm::conjugate(T), v);
	float temp = 2.0f * (v.z * T.w + v.x * T.y - T.x * v.y);
	return {
		v.x - T.y * temp,
		v.y + T.x * temp,
		temp * T.w - v.z
	};
}
//rotate by T given T.z == 0.0 and T is normalized
static inline glm::vec3 to_world(glm::quat T, glm::vec3 v) {
	//return glm::rotate(T, v);
	float temp = 2.0f * (v.z * T.w - v.x * T.y + T.x * v.y);
	return {
		v.x + T.y * temp,
		v.y - T.x * temp,
		temp * T.w - v.z
	};
}
//reflect(-v, local_normal{0,0,1})
static inline glm::vec3 reflect_N(glm::vec3 v) { return { -v.x, -v.y, v.z, }; }
//dot(local_normal{0,0,1}, v)
static inline float N_dot(glm::vec3 v) { return v.z; }
//cross(v, local_normal{0,0,1})
static inline glm::vec3 cross_N(glm::vec3 v) { return { v.y, -v.x, 0.0f }; }
//cross(local_normal{0,0,1}, v)
static inline glm::vec3 N_cross(glm::vec3 v) { return { -v.y, v.x, 0.0f }; }

/*---------------------------------
---------- Sampling Lights ---------
---------------------------------*/
float conePdf(float cosThetaMax) {
	return glm::one_over_two_pi<float>() / std::max(1e-6f, 1.0f - cosThetaMax);
}

float spherePdf(float radius_sq, float dist_sq) {
	float sinThetaMax2 = radius_sq / dist_sq;
	float cosThetaMax = sqrt(std::max(0.0f, 1.0f - sinThetaMax2));
	return conePdf(cosThetaMax);
}

static inline glm::vec3 sample_direction_within_sphere(glm::vec3 relative_light_pos, float radius2, float t, float s, float* out_distance, float* out_pdf) noexcept {
	//why not just return uniform_sphere as the sample??
	//same as with the other method we dont need a point on the light to be uniformly sampled with respect to the solid angle from P
	//but just an uniform direction from P whose usual pdf will be with respect to the solid angle
	//so this should be way simpler
	glm::vec3 light_normal = uniform_sphere(t, s);
	glm::vec3 L = relative_light_pos + light_normal * (sqrt(radius2) + 1e-4f);
	float dist2 = glm::dot(L, L);
	float dist = sqrt(dist2);
	L *= 1.0f / dist;
	float area = (4.0f * glm::pi<float>()) * radius2;
	float cos_angle = abs(glm::dot(light_normal, L));
	*out_pdf = dist2 / std::max(1e-6f, area * cos_angle);
	*out_distance = dist;
	return L;
}

//https://pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#fragment-Computeanglealphafromcenterofspheretosampledpointonsurface-0
static inline glm::vec3 sample_direction_to_sphere(const glm::vec3 Wc /*normalized*/, const float sinThetaMax2 /*(radius/center_dist)^2*/, const float center_dist, const float radius2, const float t, const float s, float* out_distance, float* out_pdf) noexcept {
	float cosThetaMax = sqrt(std::max(0.0f, 1.0f - sinThetaMax2));
	*out_pdf = conePdf(cosThetaMax);
	/* Taylor series expansion for small angles [sinThetaMax2 < sin^2(1.5 deg)], where the standard approach suffers from severe cancellation errors */
	float cosTheta = 1.0f - t * (1.0f - cosThetaMax);
	float sinTheta = sqrt(sinThetaMax2 * t);
	float src_blend = (sinThetaMax2 < 0.00068523f ? sinTheta : cosTheta);
	float invert = sqrt(std::max(0.0f, 1.0f - src_blend * src_blend));
	cosTheta = (sinThetaMax2 < 0.00068523f ? invert : cosTheta);
	sinTheta = (sinThetaMax2 < 0.00068523f ? sinTheta : invert);
	float temp = center_dist * sinTheta;
	*out_distance = center_dist * cosTheta - sqrt(std::max(0.0f, radius2 - temp * temp)) - 1e-5f; //small offset to avoid self intersection on shadow ray
	glm::vec3 Llocal = spherical_to_cartesian(s, sinTheta, cosTheta);
	glm::vec3 wcX, wcY; orthonormal_basis(Wc, &wcX, &wcY);
	return {
		wcX.x * Llocal.x + wcY.x * Llocal.y + Wc.x * Llocal.z,
		wcX.y * Llocal.x + wcY.y * Llocal.y + Wc.y * Llocal.z,
		wcX.z * Llocal.x + wcY.z * Llocal.y + Wc.z * Llocal.z
	};
}

static inline float powerHeuristic(float f, float g) noexcept {
	float f2 = f * f;
	return f2 / std::max(1e-6f, f2 + g * g);
}
static inline float powerHeuristic_over_f(float f, float g) noexcept {
	return f / std::max(1e-6f, f * f + g * g);
}

/*---------------------------------
-------------- BSDF ---------------
---------------------------------*/

//PDF = G1(V) * D(H) * max(0, H.V) / N.V;
static inline glm::vec3 distribution_visible_normals(glm::vec3 Vlocal, float alpha, float u, float v) {
	//transforming the view direction to the hemisphere configuration, that is sampling as if it was roughness = 1
	glm::vec3 V = glm::normalize(glm::vec3(alpha * Vlocal.x, alpha * Vlocal.y, Vlocal.z)); //anisotropic version would use different alpha values for x and y
	//sample a disk with each half of the disk weighted proportionally to its projection onto direction v
	auto sample = disk(u, v);
	const float t = 1.0f - sample.x * sample.x;
	sample.y = glm::mix(sqrt(t), sample.y, V.z * 0.5f + 0.5f);
	glm::vec3 X, Y; orthonormal_basis(V, &X, &Y);
	//reprojection onto hemisphere
	glm::vec3 H = X * sample.x + Y * sample.y + V * sqrt(std::max(0.0f, t - sample.y * sample.y));
	//unstretch and normalize the normal
	return glm::normalize(glm::vec3(
		alpha * H.x,
		alpha * H.y,
		std::max(0.0f, H.z)
	));
}

static inline float pow5(float x) { float t = x * x; t *= t; return x * t; }
static inline Spectrum Fresnel(Spectrum F0, float HdotV) { //or HdotL
	return glm::mix(F0, Spectrum{1.0f}, pow5(std::clamp(1.0f - HdotV, 0.0f, 1.0f)));
}

//GGX Normal distribution function
static inline float GGX_D(float alpha2, float NdotH2) {
	float temp = (1.0f + (alpha2 - 1.0f) * NdotH2);
	return alpha2 / (glm::pi<float>() * temp * temp);
}
// Smith G2 term (masking-shadowing function) for GGX distribution
// Height correlated version - optimized by substituing G_Lambda for G_Lambda_GGX and dividing by (4 * NdotL * NdotV) to cancel out 
// the terms in specular BRDF denominator
// Source: "Moving Frostbite to Physically Based Rendering" by Lagarde & de Rousiers
// Note that returned value is G2 / (4 * NdotL * NdotV) and therefore includes division by specular BRDF denominator
static inline float Smith_G2_Height_Correlated_GGX_Lagarde(float alpha2, float NdotL, float NdotV) {
	float a = NdotV * sqrt(alpha2 + NdotL * (NdotL - alpha2 * NdotL));
	float b = NdotL * sqrt(alpha2 + NdotV * (NdotV - alpha2 * NdotV));
	return 0.5f / (a + b);
}
//NdotL * (F*D*G2)/(4*NdotV*NdotL)
static inline glm::vec3 microfacet_brdf(glm::vec3 F0, float alpha, float NdotV, float NdotL, float NdotH, float HdotV) {
	const float alpha2 = alpha * alpha;
	return Fresnel(F0, HdotV) * Spectrum{NdotL * GGX_D(std::max(0.00001f, alpha2), NdotH * NdotH) * Smith_G2_Height_Correlated_GGX_Lagarde(alpha2, NdotL, NdotV)};
}
static inline float G1_GGX(float alpha2, float NdotS2) {
	return 2.0f / (1.0f + sqrt(((alpha2 * (1.0f - NdotS2)) + NdotS2) / NdotS2));
}
// G2/G1 for vndf estimator
static inline float Smith_G2_Over_G1_Height_Correlated(float alpha2, float NdotL, float NdotV) {
	float G1V = G1_GGX(alpha2, NdotV * NdotV);
	float G1L = G1_GGX(alpha2, NdotL * NdotL);
	return G1L / (G1V + G1L - G1V * G1L);
}
//F(V, Li) * G2(V, Li)/G1(V)
static inline glm::vec3 vndf_estimator(glm::vec3 F0, float alpha, float NdotV, float NdotL, float HdotV) {
	return Fresnel(F0, HdotV) * Spectrum{Smith_G2_Over_G1_Height_Correlated(alpha * alpha, NdotL, NdotV)};
}
