#pragma once

#include "VectorMath.hpp"
#include <algorithm>


//---- REFERENCE LINAR - SRGB FORMULAE----------
//// Conversion between linear and sRGB color spaces
//inline float linearToSrgb(float linearColor) {
//	if (linearColor < 0.0031308f) return linearColor * 12.92f;
//	else return 1.055f * float(pow(linearColor, 1.0f / 2.4f)) - 0.055f;
//}
//
//inline float srgbToLinear(float srgbColor) {
//	if (srgbColor < 0.04045f) return srgbColor / 12.92f;
//	else return float(pow((srgbColor + 0.055f) / 1.055f, 2.4f));
//}

INLINE float luminance(float r, float g, float b) {
	return r * 0.2126f + g * 0.7152f + b * 0.0722f;
}

INLINE Vec8f luminance(SimdVec<Vec8f, 3> rgb) {
	return mul_add(rgb[2], 0.0722f, mul_add(rgb[1], 0.7152f, rgb[0] * 0.2126f));
}

Vec8f reinhard(Vec8f x) { return x / (x + 1.0f); }
constexpr float reinhard(float x) noexcept { return x / (x + 1.0f); }

INLINE glm::vec3 ACES_input(glm::vec3 x) noexcept
{
	return {
		x.r * 0.59719f + x.g * 0.35458f + x.b * 0.04823f,
		x.r * 0.07600f + x.g * 0.90834f + x.b * 0.01566f,
		x.r * 0.02840f + x.g * 0.13383f + x.b * 0.83777f
	};
}

INLINE constexpr float ACES_rtt_odt_fit(float x) noexcept {
	return (x * (x + 0.0245786f) - 0.000090537f) / (x * (0.983729f * x + 0.4329510f) + 0.238081f);
}

INLINE glm::vec3 ACES_rtt_odt_fit(glm::vec3 x) noexcept {
	return (x * (x + 0.0245786f) - 0.000090537f) / (x * (0.983729f * x + 0.4329510f) + 0.238081f);
}

INLINE Vec8f ACES_rtt_odt_fit(Vec8f x) noexcept {
	return (x * (x + 0.0245786f) - 0.000090537f) / (x * (0.983729f * x + 0.4329510f) + 0.238081f);
}

INLINE glm::vec3 ACES_output(glm::vec3 x) noexcept {
	return {
		x.r * 1.604750f + x.g * -0.53108f + x.b * -0.07367f,
		x.r * -0.10208f + x.g *  1.10813f + x.b * -0.00605f,
		x.r * -0.00327f + x.g * -0.07276f + x.b *  1.07602f
	};
}

INLINE void tonemapping(const float r, const float g, const float b, float* out_r, float* out_g, float* out_b) noexcept {
	glm::vec3 out = ACES_output(ACES_rtt_odt_fit(ACES_input({r, g, b})));
	*out_r = std::min(1.0f, std::max(0.0f, out.r)); //std::clamp doesnt produce simple min/max assembly but tons of conditional moves and comparisons
	*out_g = std::min(1.0f, std::max(0.0f, out.g));
	*out_b = std::min(1.0f, std::max(0.0f, out.b));
}

INLINE void tonemapping(Vec8f& r, Vec8f& g, Vec8f& b) noexcept {
	Vec8f x = ACES_rtt_odt_fit(r * 0.59719f + g * 0.35458f + b * 0.04823f);
	Vec8f y = ACES_rtt_odt_fit(r * 0.07600f + g * 0.90834f + b * 0.01566f);
	Vec8f z = ACES_rtt_odt_fit(r * 0.02840f + g * 0.13383f + b * 0.83777f);
	r = min(Vec8f{1.0f}, max(Vec8f{0.0f}, x * 1.604750f + y * -0.53108f + z * -0.07367f));
	g = min(Vec8f{1.0f}, max(Vec8f{0.0f}, x * -0.10208f + y *  1.10813f + z * -0.00605f));
	b = min(Vec8f{1.0f}, max(Vec8f{0.0f}, x * -0.00327f + y * -0.07276f + z * 1.07602f));
}

INLINE uint32_t RGBA_pack_uint8(const glm::vec4& rgba_float4) {
	auto temp = _mm_load_ps((float*)&rgba_float4);
	temp = _mm_min_ps(_mm_max_ps(temp, _mm_set1_ps(0.0f)), _mm_set1_ps(1.0f));
	temp = _mm_mul_ps(temp, _mm_set_ps1(255.0f));
	return _mm_cvtsi128_si32(_mm_shuffle_epi8(_mm_cvtps_epi32(temp), _mm_cvtsi32_si128(0x0C080400)));
	//const auto RGBA_uint8 = static_cast<vec<4, uint8_t>>(clamp(rgba_float4, glm::vec4{0.0f}, glm::vec4{1.0f}) * 255.0f);
	//return RGBA_uint8.r | (RGBA_uint8.g << 8) | (RGBA_uint8.b << 16) | (RGBA_uint8.a << 24);
}

inline const auto lut_srgb_to_linear_byte = []() {
	std::array<uint8_t, 4096> ret; //tested 4096 lut improves banding over smaller luts, very close to formula
	for (size_t i = 0; i < ret.size(); i++){
		float linear = (static_cast<float>(i) + 0.5f) / static_cast<float>(ret.size() - 1);
		ret[i] = static_cast<uint8_t>(255.0f * (linear <= 0.00313066844250063 ? linear * 12.92 : 1.055 * std::pow(linear, 1.0f / 2.4f) - 0.055));
	}
	return ret;
}();

INLINE uint8_t LinearToSrgb(float linear) {
	return lut_srgb_to_linear_byte[std::min(static_cast<uint32_t>(lut_srgb_to_linear_byte.size() - 1), static_cast<uint32_t>(reinhard(std::max(0.0f, linear)) * static_cast<float>(lut_srgb_to_linear_byte.size())))];
}

INLINE uint32_t LinearToSrgb(glm::vec4 x) {
	return LinearToSrgb(x.r)
		| (LinearToSrgb(x.g) << 8)
		| (LinearToSrgb(x.b) << 16)
		| (static_cast<uint8_t>(255.0f * x.a) << 24);
}

template<Precision precision = Precision::Fast> Vec8i LinearToSrgb(Vec8f x);

template<> INLINE Vec8i LinearToSrgb<Precision::Exact>(Vec8f x) {
	x = reinhard(x);
	return roundi(select(
		x <= 0.00313066844250063,
		x * static_cast<float>(255.0 * 12.92),
		mul_sub(_mm256_pow_ps(x, _mm256_set1_ps(static_cast<float>(1.0/2.4))), static_cast<float>(255.0 * 1.055), static_cast<float>(255.0 * 0.055))
	));
}
template<> INLINE Vec8i LinearToSrgb<Precision::Fast>(Vec8f x) {
	return 0xFF & gather<1>((int32_t*)&lut_srgb_to_linear_byte, 0xFF & roundi(reinhard(x) * static_cast<float>(0xFF)));
}
template<> INLINE Vec8i LinearToSrgb<Precision::Half>(Vec8f x) {
	return roundi(sqrt<Precision::Half, 255.0f>(x) * rsqrt<Precision::Half>(x + 1.0f));
}

template<Precision precision = Precision::Fast>
INLINE Vec8i LinearToPackedSrgb(SimdVec<Vec8f, 3> linear_color) {
	return _mm256_shuffle_epi8(								//RGBA'RGBA'RGBA'RGBA'RGBA'RGBA'RGBA'RGBA
		_mm256_packus_epi16(								//RRRR'GGGG'BBBB'AAAA'RRRR'GGGG'BBBB'AAAA
			_mm256_packus_epi32(							//.R.R'.R.R'.G.G'.G.G'.R.R'.R.R'.G.G'.G.G
				LinearToSrgb<precision>(linear_color[0]),	//...R'...R'...R'...R'...R'...R'...R'...R
				LinearToSrgb<precision>(linear_color[1])),	//...G'...G'...G'...G'...G'...G'...G'...G
			_mm256_packus_epi32(				            //.B.B'.B.B'.A.A'.A.A'.B.B'.B.B'.A.A'.A.A
				LinearToSrgb<precision>(linear_color[2]),	//...B'...B'...B'...B'...B'...B'...B'...B
				_mm256_set1_epi32(0xFF))),		            //...A'...A'...A'...A'...A'...A'...A'...A
		_mm256_setr_epi8(0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15,0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15));
}