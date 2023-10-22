#pragma once

#include <functional>

#include "Core.hpp"

template <size_t Bits> struct UnsignedInt {};
template <> struct UnsignedInt< 8> { using Type = uint8_t ; };
template <> struct UnsignedInt<16> { using Type = uint16_t; };
template <> struct UnsignedInt<32> { using Type = uint32_t; };
template <> struct UnsignedInt<64> { using Type = uint64_t; };
template <size_t Bits> using UnsignedIntType = typename UnsignedInt<Bits>::Type;

template <size_t Bits> struct SignedInt {};
template <> struct SignedInt< 8> { using Type = int8_t ; };
template <> struct SignedInt<16> { using Type = int16_t; };
template <> struct SignedInt<32> { using Type = int32_t; };
template <> struct SignedInt<64> { using Type = int64_t; };
template <size_t Bits> using SignedIntType = typename SignedInt<Bits>::Type;


template <typename, size_t> struct simd_register {};
template <> struct simd_register<float, 4> { using Type = Vec4f ; };
template <> struct simd_register<float, 8> { using Type = Vec8f; };
template <> struct simd_register<double, 2> { using Type = Vec2d; };
template <> struct simd_register<double, 4> { using Type = Vec4d; };
template <> struct simd_register<int8_t, 16> { using Type = Vec16c ; };
template <> struct simd_register<int8_t, 32> { using Type = Vec32c; };
template <> struct simd_register<int16_t, 8> { using Type = Vec8s; };
template <> struct simd_register<int16_t, 16> { using Type = Vec16s; };
template <> struct simd_register<int32_t, 4> { using Type = Vec4i ; };
template <> struct simd_register<int32_t, 8> { using Type = Vec8i; };
template <> struct simd_register<int64_t, 2> { using Type = Vec2q; };
template <> struct simd_register<int64_t, 4> { using Type = Vec4q; };
template <> struct simd_register<uint8_t, 16> { using Type = Vec16uc ; };
template <> struct simd_register<uint8_t, 32> { using Type = Vec32uc; };
template <> struct simd_register<uint16_t, 8> { using Type = Vec8us; };
template <> struct simd_register<uint16_t, 16> { using Type = Vec16us; };
template <> struct simd_register<uint32_t, 4> { using Type = Vec4ui ; };
template <> struct simd_register<uint32_t, 8> { using Type = Vec8ui; };
template <> struct simd_register<uint64_t, 2> { using Type = Vec2uq; };
template <> struct simd_register<uint64_t, 4> { using Type = Vec4uq; };


template <typename T, size_t N> using simd_register_t = typename simd_register<T, N>::Type;


template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
constexpr T make_bitmask(size_t bits) {
	return bits >= std::numeric_limits<T>::digits ? static_cast<T>(-1) : (static_cast<T>(1) << bits) - 1;
}

// Return the second argument if the first one is a NaN.
template <std::floating_point fp_type> fp_type robust_min(fp_type a, fp_type b) { return a < b ? a : b; }
template <std::floating_point fp_type> fp_type robust_max(fp_type a, fp_type b) { return a > b ? a : b; }
/// Adds the given number of ULPs (Units in the Last Place) to the given floating-point number.
template <std::floating_point fp_type> fp_type add_ulp_magnitude(fp_type x, unsigned ulps) {
	return std::bit_cast<fp_type>(std::bit_cast<UnsignedIntType<sizeof(fp_type) * CHAR_BIT>>(x) + ulps);
}
// Inverse of the given value, always returning a finite value.
template <std::floating_point fp_type> fp_type safe_inverse(fp_type x) {
	return std::fabs(x) <= std::numeric_limits<fp_type>::epsilon()
		? std::copysign(std::numeric_limits<fp_type>::max(), x)
		: static_cast<fp_type>(1.0) / x;
}

Vec8f add_ulp_magnitude(Vec8f x, unsigned ulps) {
	return _mm256_castsi256_ps(Vec8i{_mm256_castps_si256(x)} + ulps);
}

Vec8f safe_inverse(Vec8f x) {
	return select(
		abs(x) <= std::numeric_limits<float>::epsilon(),
		sign_combine(std::numeric_limits<float>::max(), x),
		static_cast<float>(1.0) / x
	);
}

enum class Precision {
	Exact,
	Fast, //2^-22 to 2^-23
	Half //2^-11 up to 2^-14
};

template<typename T, size_t N> struct SimdVec {
	using value_type = T;
	value_type _data[N];
	template <typename Self> inline auto& operator[](this Self&& self, size_t i) { return self._data[i]; }
};
template<typename T> struct SimdVec<T,2> {
	using value_type = T;
	value_type x, y;
	template <typename Self> inline auto& operator[](this Self&& self, size_t i) { return (&(self.x))[i]; }
};
template<typename T> struct SimdVec<T, 3> {
	using value_type = T;
	value_type x, y, z;
	template <typename Self> inline auto& operator[](this Self&& self, size_t i) { return (&(self.x))[i]; }
};
template<typename T> struct SimdVec<T, 4> {
	using value_type = T;
	value_type x, y, z, w;
	template <typename Self> inline auto& operator[](this Self&& self, size_t i) { return (&(self.x))[i]; }
};
static_assert(sizeof(SimdVec<Vec8f, 1>) == sizeof(Vec8f) * 1);
static_assert(sizeof(SimdVec<Vec8f, 2>) == sizeof(Vec8f) * 2);
static_assert(sizeof(SimdVec<Vec8f, 3>) == sizeof(Vec8f) * 3);
static_assert(sizeof(SimdVec<Vec8f, 4>) == sizeof(Vec8f) * 4);


using Vec8x2f = SimdVec<Vec8f, 2>;
using Vec8x3f = SimdVec<Vec8f, 3>;
using Vec8x4f = SimdVec<Vec8f, 4>;

#define MAKEOP(OP)\
template<typename T, size_t N> INLINE SimdVec<T, N>& operator##OP##=(SimdVec<T, N>& in_place, SimdVec<T, N> operand) { [&] <size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> void { ((in_place[i] OP##= operand[i]), ...); }(std::make_index_sequence<N>{}); return in_place; }\
template<typename T, size_t N> INLINE SimdVec<T, N>& operator##OP##=(SimdVec<T, N>& in_place, T operand) { [&] <size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> void { ((in_place[i] OP##= operand), ...); }(std::make_index_sequence<N>{}); return in_place; }\
template<typename T, size_t N> INLINE SimdVec<T, N>& operator##OP##=(SimdVec<T, N>& in_place, glm::vec<N, float> operand_) { const float* operand = (float*)&operand_; [&] <size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> void { ((in_place[i] OP##= operand[i]), ...); }(std::make_index_sequence<N>{}); return in_place; }\
template<typename T, size_t N> INLINE SimdVec<T, N>  operator##OP##(SimdVec<T, N> in_place, SimdVec<T, N> operand) { return in_place OP##= operand; }\
template<typename T, size_t N> INLINE SimdVec<T, N>  operator##OP##(SimdVec<T, N> in_place, T operand) { return in_place OP##= operand; }\
template<typename T, size_t N> INLINE SimdVec<T, N>  operator##OP##(SimdVec<T, N> in_place, glm::vec<N, float> operand) { return in_place OP##= operand; }\
template<typename T, size_t N> INLINE SimdVec<T, N>  operator##OP##(T operand, SimdVec<T, N> in_place) { [&] <size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> void { ((in_place[i] = operand OP in_place[i]), ...); }(std::make_index_sequence<N>{}); return in_place; }\
template<typename T, size_t N> INLINE SimdVec<T, N>  operator##OP##(glm::vec<N, float> operand, SimdVec<T, N> in_place) { [&] <size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> void { ((in_place[i] = operand[i] OP in_place[i]), ...); }(std::make_index_sequence<N>{}); return in_place; }
MAKEOP(+); 	MAKEOP(-); 	MAKEOP(*); 	MAKEOP(/); 	MAKEOP(%); MAKEOP(&);
#undef MAKEOP

template<size_t N> INLINE SimdVec<Vec8f, N> operator*(glm::vec<N, float> a, Vec8f b) {
	return[&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<Vec8f, N> { return {(a[i] * b)...}; }(std::make_index_sequence<N>{});
}
template<size_t N> INLINE SimdVec<Vec8f, N> operator*(Vec8f a, glm::vec<N, float> b) {
	return b * a;
}
template<size_t N> INLINE SimdVec<Vec8f, N> operator+(glm::vec<N, float> a, Vec8f b) {
	return[&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<Vec8f, N> { return {(a[i] + b)...}; }(std::make_index_sequence<N>{});
}
template<size_t N> INLINE SimdVec<Vec8f, N> operator+(Vec8f a, glm::vec<N, float> b) {
	return b + a;
}

INLINE Vec8f nmul_sub(Vec8f a, Vec8f b, Vec8f c) {
	return _mm256_fnmsub_ps(a, b, c);
}

template<typename T, size_t N> INLINE SimdVec<T, N> mul_add(SimdVec<T, N> a, SimdVec<T, N>b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(mul_add(a[i], b[i], c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> mul_add(T a, SimdVec<T, N>b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(mul_add(a, b[i], c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> mul_add(SimdVec<T, N> a, T b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(mul_add(a[i], b, c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> mul_add(SimdVec<T, N> a, SimdVec<T, N>b, T c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(mul_add(a[i], b[i], c))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> mul_sub(SimdVec<T, N> a, SimdVec<T, N>b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(mul_sub(a[i], b[i], c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> mul_sub(T a, SimdVec<T, N>b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(mul_sub(a, b[i], c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> mul_sub(SimdVec<T, N> a, T b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(mul_sub(a[i], b, c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> mul_sub(SimdVec<T, N> a, SimdVec<T, N>b, T c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(mul_sub(a[i], b[i], c))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> nmul_add(SimdVec<T, N> a, SimdVec<T, N>b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(nmul_add(a[i], b[i], c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> nmul_add(T a, SimdVec<T, N>b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(nmul_add(a, b[i], c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> nmul_add(SimdVec<T, N> a, T b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(nmul_add(a[i], b, c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> nmul_add(SimdVec<T, N> a, SimdVec<T, N>b, T c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(nmul_add(a[i], b[i], c))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> nmul_sub(SimdVec<T, N> a, SimdVec<T, N>b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(nmul_sub(a[i], b[i], c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> nmul_sub(T a, SimdVec<T, N>b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(nmul_sub(a, b[i], c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> nmul_sub(SimdVec<T, N> a, T b, SimdVec<T, N> c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(nmul_sub(a[i], b, c[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> nmul_sub(SimdVec<T, N> a, SimdVec<T, N>b, T c) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(nmul_sub(a[i], b[i], c))...}; }(std::make_index_sequence<N>{}); }

template<typename T, size_t N> INLINE T hsum(const SimdVec<T, N> a) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> T { return (a[i] + ...); }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> min(SimdVec<T, N> a, SimdVec<T, N>b) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(min(a[i], b[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> max(SimdVec<T, N> a, SimdVec<T, N>b) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> { return {(max(a[i], b[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> clamp(SimdVec<T, N> a, float lower, float upper) { for (size_t i = 0; i < N; i++) a[i] = min(max(a[i], lower), upper); return a; }
template<typename T, size_t N> INLINE SimdVec<T, N> pow(SimdVec<T, N> a, float exponent) { for (size_t i = 0; i < N; i++) a[i] = pow(a[i], exponent); return a; }
template<typename T, size_t N> INLINE T dot(const SimdVec<T, N> a, const SimdVec<T, N> b) { return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> T { return ((a[i] * b[i]) + ...); }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE T dot(const glm::vec<N, float>& a_, const SimdVec<T, N> b) { const float* a = (float*)&a_; return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> T { return ((a[i] * b[i]) + ...); }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE T dot(const SimdVec<T, N> a, const glm::vec<N, float>& b_) { const float* b = (float*)&b_; return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> T { return ((a[i] * b[i]) + ...); }(std::make_index_sequence<N>{}); }
template<Precision precision = Precision::Fast, typename T, size_t N> INLINE SimdVec<T, N> normalize(const SimdVec<T, N> a) { return a * rsqrt<precision>(dot(a, a)); }
template<size_t N> INLINE SimdVec<Vec8f, N> select(Vec8fb mask, const SimdVec<Vec8f, N> a, const SimdVec<Vec8f, N> b){ return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<Vec8f, N> { return {(select(mask, a[i], b[i]))...}; }(std::make_index_sequence<N>{}); }
template<typename T, size_t N> INLINE SimdVec<T, N> reflect(const SimdVec<T, N> a, const SimdVec<T, N> b) {
	return [&]<size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> SimdVec<T, N> {
		T temp = dot(b, a); temp += temp;
		return {nmul_add(b[i], temp, a[i])...};
	}(std::make_index_sequence<N>{});
}
template<typename T> INLINE SimdVec<T, 3> cross(const SimdVec<T, 3> a, const SimdVec<T, 3> b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}
template<> INLINE Vec8f dot<Vec8f, 3>(const SimdVec<Vec8f, 3> a, const SimdVec<Vec8f, 3> b) {
	return _mm256_fmadd_ps(a[2], b[2], _mm256_fmadd_ps(a[1], b[1], a[0] * b[0]));
}
INLINE Vec8f ndot(const SimdVec<Vec8f, 3> a, const SimdVec<Vec8f, 3> b) {
	return _mm256_fnmsub_ps(a[2], b[2], _mm256_fmadd_ps(a[1], b[1], a[0] * b[0]));
}

INLINE Vec8f max_component(const SimdVec<Vec8f, 3> a) {
	return max(max(a[0], a[1]), a[2]);
}
INLINE Vec8f min_component(const SimdVec<Vec8f, 3> a) {
	return min(min(a[0], a[1]), a[2]);
}
INLINE Vec8f max_component(const SimdVec<Vec8f, 4> a) {
	return max(max(max(a[0], a[1]), a[2]), a[3]);
}
INLINE Vec8f min_component(const SimdVec<Vec8f, 4> a) {
	return min(min(min(a[0], a[1]), a[2]), a[3]);
}

INLINE Vec8f lerp(Vec8f a, Vec8f b, Vec8f t) {
	return mul_add(b - a, t, a); //(1 - t) * a + t * b
}

INLINE bool is_broadcast(Vec8i a) {
	return _mm256_movemask_epi8(_mm256_cmpeq_epi32(a, _mm256_broadcastd_epi32(_mm256_castsi256_si128(a)))) == 0xffffffffU;
}

template<size_t scale = sizeof(float)> requires(scale == 1 || scale == 2 || scale == 4 || scale == 8)
INLINE Vec8f gather(const float* ptr, Vec8i indices) {
	return _mm256_i32gather_ps(ptr, indices, scale);
}
template<size_t scale = sizeof(int32_t)> requires(scale == 1 || scale == 2 || scale == 4 || scale == 8)
INLINE Vec8i gather(const int32_t* ptr, Vec8i indices) {
	return _mm256_i32gather_epi32(ptr, indices, scale);
}

template<size_t scale = sizeof(float)> requires(scale == 1 || scale == 2 || scale == 4 || scale == 8)
INLINE Vec8f gather(const float* ptr, Vec8i indices, Vec8f src, Vec8fb mask) {
	return _mm256_mask_i32gather_ps(src, ptr, indices, mask, scale);
}
template<size_t scale = sizeof(int32_t)> requires(scale == 1 || scale == 2 || scale == 4 || scale == 8)
INLINE Vec8i gather(const int32_t* ptr, Vec8i indices, Vec8i src, Vec8ib mask) {
	return _mm256_mask_i32gather_epi32(src, ptr, indices, mask, scale);
}

INLINE int32_t mask_compress(Vec8fb mask) {
	return _mm256_movemask_ps(mask);
}

INLINE Vec8fb mask_expand(int32_t bits) {
	const Vec8i bitmask{1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7};
	return (_mm256_set1_epi32(bits) & bitmask) == bitmask;
}

template<int32_t bits, typename vector_type, std::integral scalar_type = decltype(vector_type{}[0])>
INLINE vector_type constant_lshift(vector_type x) {
	if constexpr (bits == 0) return x;
	static constexpr size_t width = sizeof(scalar_type);
	static constexpr size_t register_size = sizeof(vector_type);
	if constexpr (register_size == 32) {
		if constexpr (bits > 0) {
			if constexpr (width == 2) return _mm256_slli_epi16(x, bits);
			if constexpr (width == 4) return _mm256_slli_epi32(x, bits);
			if constexpr (width == 8) return _mm256_slli_epi64(x, bits);
		} else {
			if constexpr (width == 2) return _mm256_srli_epi16(x, -bits);
			if constexpr (width == 4) return _mm256_srli_epi32(x, -bits);
			if constexpr (width == 8) return _mm256_srli_epi64(x, -bits);
		}
	}
	if constexpr (register_size == 16) {
		if constexpr (bits > 0) {
			if constexpr (width == 2) return _mm_slli_epi16(x, bits);
			if constexpr (width == 4) return _mm_slli_epi32(x, bits);
			if constexpr (width == 8) return _mm_slli_epi64(x, bits);
		} else {
			if constexpr (width == 2) return _mm_srli_epi16(x, -bits);
			if constexpr (width == 4) return _mm_srli_epi32(x, -bits);
			if constexpr (width == 8) return _mm_srli_epi64(x, -bits);
		}
	}
}

template<Precision precision = Precision::Fast, bool reciprocal = false, float scalar = 1.0f> Vec8f sqrt_impl(Vec8f x) {
	if constexpr (precision == Precision::Exact)
		return std::conditional_t<reciprocal, std::divides<Vec8f>, std::multiplies<Vec8f>>{}(scalar, _mm256_sqrt_ps(x));
	Vec8f one_over_aproximation = approx_rsqrt(x); Vec8f aproximation = x * one_over_aproximation;
	Vec8f ret = (reciprocal ? one_over_aproximation : aproximation);
	ret *= (precision == Precision::Half ? scalar : 0.5f * scalar);
	if constexpr (precision == Precision::Fast) 
		ret *= nmul_add(aproximation, one_over_aproximation, 3.0f);
	return ret;
}
template<Precision precision = Precision::Fast, float scalar = 1.0f> INLINE Vec8f sqrt(Vec8f x) { return sqrt_impl<precision, false, scalar>(x); }
template<Precision precision = Precision::Fast, float scalar = 1.0f> INLINE Vec8f rsqrt(Vec8f x) { return sqrt_impl<precision, true, scalar>(x); }

template<Precision precision = Precision::Fast, float scalar = 1.0f> Vec8f recipr(Vec8f x) {
	if constexpr (precision == Precision::Exact) return scalar / x;
	Vec8f aproximation = approx_recipr(x);
	Vec8f ret = scalar * aproximation;
	if constexpr (precision == Precision::Fast) ret *= nmul_add(aproximation, x, 2.0f);
	return ret;
}

enum class ValueAssumption {
	Null = 0,
	Normalized = 1, //-1 <= x <= 1
	Positive = 2,   //x >= 0
	Negative = 4,   //x < 0
	Lower = 8,      //abs(x) < 0.5
	Upper = 16,      //abs(x) > 0.5

	PositiveLower = Positive | Lower,
	PositiveUpper = Positive | Upper,
	NegativeLower = Negative | Lower,
	NegativeUpper = Negative | Upper,
	NormalizedLower = Normalized | Lower,
	NormalizedUpper = Normalized | Upper,
	NormalizedPositiveLower = Normalized | Positive | Lower,
	NormalizedPositiveUpper = Normalized | Positive | Upper,
	NormalizedNegativeLower = Normalized | Negative | Lower,
	NormalizedNegativeUpper = Normalized | Negative | Upper,
};
consteval bool operator!(ValueAssumption a) { return !static_cast<int>(a); }
consteval ValueAssumption operator|(ValueAssumption a, ValueAssumption b) { return ValueAssumption{static_cast<int>(a) | static_cast<int>(b)}; }
consteval ValueAssumption operator&(ValueAssumption a, ValueAssumption b) { return ValueAssumption{static_cast<int>(a) & static_cast<int>(b)}; }

template<ValueAssumption range = ValueAssumption::Normalized> requires (!!(range& ValueAssumption::Normalized))
INLINE Vec8f acos(Vec8f x) {
	static constexpr float pi = 3.14159265358979323846264338327950288f;
	static constexpr float half_pi = 1.57079632679489661923132169163975144f;
	Vec8f xa, x3, x4;
	if constexpr (!!(range & ValueAssumption::Positive)) { 
		xa = x;
	} else if constexpr (!!(range & ValueAssumption::Negative)) {
		xa = -x;
	} else /*constexpr*/ { 
		xa = abs(x); 
	}
	if constexpr (!!(range & ValueAssumption::Lower)) {
		x3 = xa * xa;
		x4 = xa;
	} else /*constexpr*/ {
		Vec8f temp = nmul_add(xa, 0.5f, 0.5f);
		if constexpr (!!(range & ValueAssumption::Upper)) {
			x3 = temp;
			x4 = sqrt<Precision::Half>(temp);
		} else /*constexpr*/ {
			x3 = select(xa > 0.5f, temp, xa * xa);
			x4 = select(xa > 0.5f, sqrt<Precision::Half>(temp), xa);
		}
	}
	Vec8f z = mul_add(polynomial_4(x3, 1.6666752422E-1f, 7.4953002686E-2f, 4.5470025998E-2f, 2.4181311049E-2f, 4.2163199048E-2f), x3 * x4, x4);
	if constexpr (!!(range & ValueAssumption::Lower)) {
		return half_pi - sign_combine(z, x);
	} else /*constexpr*/ {
		Vec8f temp = z + z;
		if constexpr (!!(range & ValueAssumption::Positive)) {
			if constexpr (!!(range & ValueAssumption::Upper)) {
				return temp;
			} else /*constexpr*/ {
				return select(xa > 0.5f, temp, half_pi - sign_combine(z, x));
			}
		} else {
			Vec8f sign_correct = select(x < 0.0f, pi - temp, temp);
			if constexpr (!!(range & ValueAssumption::Upper)) {
				return sign_correct;
			} else /*constexpr*/ {
				return select(xa > 0.5f, sign_correct, half_pi - sign_combine(z, x));
			}
		}
	}
}
struct sincosret { Vec8f sin; Vec8f cos; };
INLINE sincosret sin_cos_zero_twopi(const Vec8f xx) {
	// Find quadrant
	//      0 -   pi/4 => 0
	//   pi/4 - 3*pi/4 => 1
	// 3*pi/4 - 5*pi/4 => 2
	// 5*pi/4 - 7*pi/4 => 3
	// 7*pi/4 - 8*pi/4 => 4
	Vec8f y = round(xx * 2.0f / 3.14159265358979323846264338327950288f);// quadrant, as float
	Vec8i q = roundi(y);              // quadrant, as integer
	// Reduce by extended precision modular arithmetic
	Vec8f x = nmul_add(y, 3.77489497744594108E-8f * 2.f, nmul_add(y, 2.4187564849853515625E-4f * 2.f, nmul_add(y, 0.78515625f * 2.f, xx)));
	// Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
	Vec8f x2 = x  * x, x3 = x2 * x, x4 = x2 * x2;
	Vec8f s = mul_add(mul_add(x4, -1.9515295891E-4f, mul_add(x2, 8.3321608736E-3f, -1.6666654611E-1f)), x3, x);
	Vec8f c = mul_add(mul_add(x4, 2.443315711809948E-5f, mul_add(x2, -1.388731625493765E-3f, 4.166664568298827E-2f)), x4, nmul_add(0.5f, x2, 1.0f));
	// swap sin and cos if odd quadrant
	Vec8fb swap = Vec8fb((q & 1) != 0);
	return {
		sign_combine(select(swap, c, s), reinterpret_f(((q << 30) ^ Vec8i(reinterpret_i(xx))))),
		select(swap, s, c) ^ reinterpret_f(((q + 1) & 2) << 30)
	};
}
//input is assumed to be in range 0-1 and sin cos are returned for input * 2*pi
INLINE sincosret sin_cos_zero_one(const Vec8f xx) {
	// Find quadrant
	//      0 -   pi/4 => 0
	//   pi/4 - 3*pi/4 => 1
	// 3*pi/4 - 5*pi/4 => 2
	// 5*pi/4 - 7*pi/4 => 3
	// 7*pi/4 - 8*pi/4 => 4
	Vec8f y = round(xx * 4.0f);// quadrant, as float
	Vec8i q = roundi(y);              // quadrant, as integer
	// Reduce by extended precision modular arithmetic
	Vec8f x = nmul_add(y, 3.77489497744594108E-8f * 2.f, nmul_add(y, 2.4187564849853515625E-4f * 2.f, nmul_add(y, 0.78515625f * 2.f, xx * glm::two_pi<float>())));
	// Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
	Vec8f x2 = x * x, x3 = x2 * x, x4 = x2 * x2;
	Vec8f s = mul_add(mul_add(x4, -1.9515295891E-4f, mul_add(x2, 8.3321608736E-3f, -1.6666654611E-1f)), x3, x);
	Vec8f c = mul_add(mul_add(x4, 2.443315711809948E-5f, mul_add(x2, -1.388731625493765E-3f, 4.166664568298827E-2f)), x4, nmul_add(0.5f, x2, 1.0f));
	// swap sin and cos if odd quadrant
	Vec8fb swap = Vec8fb((q & 1) != 0);
	return {
		sign_combine(select(swap, c, s), reinterpret_f(((q << 30) ^ Vec8i(reinterpret_i(xx))))),
		select(swap, s, c) ^ reinterpret_f(((q + 1) & 2) << 30)
	};
}

template<Precision precision = Precision::Fast, typename T, size_t N> INLINE T norm(const SimdVec<T, N>& a) { return sqrt<precision>(dot(a, a)); }
template<Precision precision = Precision::Fast, typename T, size_t N> INLINE T rnorm(const SimdVec<T, N>& a) { return rsqrt<precision>(dot(a, a)); }

template<size_t N, typename CoefType> INLINE Vec8f polynomial(Vec8f x, const CoefType coeffs[N]) {
	Vec8f res = coeffs[0];
	[&] <size_t... i>(std::index_sequence<i...>) [[msvc::forceinline]] -> void {
		((res = mul_add(res, x, Vec8f{coeffs[i + 1]})), ...);
	}(std::make_index_sequence<N - 1>{});
	return res;
}

/*			
A) Posicionar en la horizontal correcta cada elemento con permutes
B) blend every third -> 0b[...]1001[...] patterns, grabbing diagonals through the matrix
[x0,x1,x2,x3|x4,x5,x6,x7] => permute8<0,3,6,1,4,7,2,5> => [x0,x3,x6,x1|x4,x7,x2,x5] => blend<x[10010010] | y[01001001] | z[00100100]> => [x0,y0,z0,x1|y1,z1,x2,y2]
[y0,y1,y2,y3|y4,y5,y6,y7] => permute8<5,0,3,6,1,4,7,2> => [y5,y0,y3,y6|y1,y4,y7,y2] => blend<x[01001001] | y[00100100] | z[10010010]> => [z2,x3,y3,z3|x4,y4,z4,x5]
[z0,z1,z2,z3|z4,z5,z6,z7] => permute8<2,5,0,3,6,1,4,7> => [z2,z5,z0,z3|z6,z1,z4,z7] => blend<x[00100100] | y[10010010] | z[01001001]> => [y5,z5,x6,y6|z6,x7,y7,z7]

 [x0,y0,z0,x1|y1,z1,x2,y2] => same blends => [x0,x3,x6,x1|x4,x7,x2,x5] => permute8<0,3,6,1,4,7,2,5> => [x0,x1,x2,x3|x4,x5,x6,x7]
 [z2,x3,y3,z3|x4,y4,z4,x5] => same blends => [y5,y0,y3,y6|y1,y4,y7,y2] => permute8<1,4,7,2,5,0,3,6> => [y0,y1,y2,y3|y4,y5,y6,y7]
 [y5,z5,x6,y6|z6,x7,y7,z7] => same blends => [z2,z5,z0,z3|z6,z1,z4,z7] => permute8<2,5,0,3,6,1,4,7> => [z0,z1,z2,z3|z4,z5,z6,z7]
*/

INLINE SimdVec<Vec8f, 3> SoA_load(const float src[8 * 3]) {
	__m256 a = _mm256_load_ps(&src[0 ]);
	__m256 b = _mm256_load_ps(&src[8 ]);
	__m256 c = _mm256_load_ps(&src[16]);
	return {
		permute8<0,3,6,1,4,7,2,5>(_mm256_blend_ps(_mm256_blend_ps(a, b, 0b01001001), c, 0b00100100)),
		permute8<1,4,7,2,5,0,3,6>(_mm256_blend_ps(_mm256_blend_ps(a, b, 0b00100100), c, 0b10010010)),
		permute8<2,5,0,3,6,1,4,7>(_mm256_blend_ps(_mm256_blend_ps(a, b, 0b10010010), c, 0b01001001))
	};
}

INLINE void Aos_store(const SimdVec<Vec8f, 3> src, float dst[8 * 3]) {
	__m256 x = permute8<0,3,6,1,4,7,2,5>(src[0]);
	__m256 y = permute8<5,0,3,6,1,4,7,2>(src[1]);
	__m256 z = permute8<2,5,0,3,6,1,4,7>(src[2]);
	*(__m256*)&dst[0 ] = _mm256_blend_ps(_mm256_blend_ps(x, y, 0b01001001), z, 0b00100100);
	*(__m256*)&dst[8 ] = _mm256_blend_ps(_mm256_blend_ps(x, y, 0b00100100), z, 0b10010010);
	*(__m256*)&dst[16] = _mm256_blend_ps(_mm256_blend_ps(x, y, 0b10010010), z, 0b01001001);
}

INLINE Vec16f mat_mul(const float ma[16], const float mb[16]) {
	const Vec8f a00 = _mm256_broadcast_ps((const __m128*) & ma[0]);
	const Vec8f a11 = _mm256_broadcast_ps((const __m128*) & ma[4]);
	const Vec8f a22 = _mm256_broadcast_ps((const __m128*) & ma[8]);
	const Vec8f a33 = _mm256_broadcast_ps((const __m128*) & ma[12]);
	const Vec8f b01 = *(Vec8f*)&mb[0];
	const Vec8f b23 = *(Vec8f*)&mb[8];
	return {
		mul_add(a00,  permute8<0,0,0,0,4,4,4,4>(b01),
		mul_add(a11,  permute8<1,1,1,1,5,5,5,5>(b01),
		mul_add(a22,  permute8<2,2,2,2,6,6,6,6>(b01),
				a33 * permute8<3,3,3,3,7,7,7,7>(b01)
		))),
		mul_add(a00,  permute8<0,0,0,0,4,4,4,4>(b23),
		mul_add(a11,  permute8<1,1,1,1,5,5,5,5>(b23),
		mul_add(a22,  permute8<2,2,2,2,6,6,6,6>(b23),
				a33 * permute8<3,3,3,3,7,7,7,7>(b23)
		)))
	};
}

INLINE Vec16f mat_transpose(const float m[16]) {
	Vec8f t0 = *(const Vec8f*)&m[0];
	Vec8f t1 = *(const Vec8f*)&m[8];
	t0 = _mm256_shuffle_ps(t0, t0, 0b11'01'10'00);
	t1 = _mm256_shuffle_ps(t1, t1, 0b11'01'10'00);
	t0 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(t0), 0b11'01'10'00));
	t1 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(t1), 0b11'01'10'00));
	return {
		_mm256_shuffle_ps(t0, t1, 0b10'00'10'00),
		_mm256_shuffle_ps(t0, t1, 0b11'01'11'01)
	};
}

__m256 compress(__m256 src, unsigned int mask /* from movmskps */)
{
	uint64_t expanded_mask = 0xFF * _pdep_u64(mask, 0x0101010101010101);  // unpack each bit smeared to a byte
	uint64_t indices = _pext_u64(0x0706050403020100/*identiy shuffle*/, expanded_mask);
	return _mm256_permutevar8x32_ps(src, _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(indices)));
}

__m256 expand(__m256 src, unsigned int mask /* from movmskps */)
{
	uint64_t expanded_mask = 0xFF * _pdep_u64(mask, 0x0101010101010101);  // unpack each bit smeared to a byte
	uint64_t indices = _pdep_u64(0x0706050403020100/*identiy shuffle*/, expanded_mask);
	return _mm256_permutevar8x32_ps(src, _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(indices)));
}

INLINE Vec8x4f qsolve(Vec8x3f from, Vec8x3f to) {
	const Vec8x3f H = normalize(from + to);
	return {
		from.y * H.z - from.z * H.y,
		from.z * H.x - from.x * H.z,
		from.x * H.y - from.y * H.x,
		dot(from, H)
	};
}

INLINE glm::vec3 qrotate(glm::quat q, glm::vec3 v) {
	const glm::vec3  crossprod{
		q.y * v.z - q.z * v.y,
		q.z * v.x - q.x * v.z,
		q.x * v.y - q.y * v.x
	};
	return {
		v.x + 2.0f * (q.w * crossprod.x + q.y * crossprod.z - q.z * crossprod.y),
		v.y + 2.0f * (q.w * crossprod.y + q.z * crossprod.x - q.x * crossprod.z),
		v.z + 2.0f * (q.w * crossprod.z + q.x * crossprod.y - q.y * crossprod.x)
	};
}

INLINE Vec8x3f qrotate(glm::quat q, Vec8x3f v) {
	const Vec8x3f crossprod{
		mul_sub(q.y, v.z, q.z * v.y),
		mul_sub(q.z, v.x, q.x * v.z),
		mul_sub(q.x, v.y, q.y * v.x)
	};
	return {
		v.x + 2.0f * (q.w * crossprod.x + q.y * crossprod.z - q.z * crossprod.y),
		v.y + 2.0f * (q.w * crossprod.y + q.z * crossprod.x - q.x * crossprod.z),
		v.z + 2.0f * (q.w * crossprod.z + q.x * crossprod.y - q.y * crossprod.x)
	};
}


INLINE Vec8x3f qrotate(Vec8x4f q, Vec8x3f v) {
	const Vec8x3f crossprod{
		mul_sub(q.y, v.z, q.z * v.y),
		mul_sub(q.z, v.x, q.x * v.z),
		mul_sub(q.x, v.y, q.y * v.x)
	};
	return {
		v.x + 2.0f * (q.w * crossprod.x + q.y * crossprod.z - q.z * crossprod.y),
		v.y + 2.0f * (q.w * crossprod.y + q.z * crossprod.x - q.x * crossprod.z),
		v.z + 2.0f * (q.w * crossprod.z + q.x * crossprod.y - q.y * crossprod.x)
	};
}

INLINE Vec8x3f qrotate_conjugate(Vec8x4f q, Vec8x3f v) {
	const Vec8x3f crossprod{
		mul_sub(q.z, v.y, q.y * v.z),
		mul_sub(q.x, v.z, q.z * v.x),
		mul_sub(q.y, v.x, q.x * v.y)
	};
	return {
		v.x + 2.0f * (q.w * crossprod.x - q.y * crossprod.z + q.z * crossprod.y),
		v.y + 2.0f * (q.w * crossprod.y - q.z * crossprod.x + q.x * crossprod.z),
		v.z + 2.0f * (q.w * crossprod.z - q.x * crossprod.y + q.y * crossprod.x)
	};
}

INLINE Vec8x3f nqrotate_conjugate(Vec8x4f q, Vec8x3f v) {
	const Vec8x3f crossprod{
		mul_sub(q.y, v.z, q.z * v.y),
		mul_sub(q.z, v.x, q.x * v.z),
		mul_sub(q.x, v.y, q.y * v.x)
	};
	return {
		2.0f * (q.w * crossprod.x - q.y * crossprod.z + q.z * crossprod.y) - v.x,
		2.0f * (q.w * crossprod.y - q.z * crossprod.x + q.x * crossprod.z) - v.y,
		2.0f * (q.w * crossprod.z - q.x * crossprod.y + q.y * crossprod.x) - v.z
	};
}

template<std::unsigned_integral T>
static inline T isolate_lsb(const T val) noexcept {
	if constexpr (sizeof(T) <= 4) {
		return static_cast<T>(_blsi_u32(val));
	} else {
		return static_cast<T>(_blsi_u64(val));
	}
}

static inline float fast_abs(float value) noexcept {
	const __m128 sign_mask = _mm_set_ss(-0.0f);
	return _mm_cvtss_f32(_mm_andnot_ps(sign_mask, _mm_set_ss(value)));
}
static inline float fast_sign(float sign) noexcept { //copysign(1.0f, x)
	const __m128 sign_mask = _mm_set_ss(-0.0f);
	return _mm_cvtss_f32(_mm_or_ps(_mm_set_ss(1.0f), _mm_and_ps(sign_mask, _mm_set_ss(sign))));
}
static inline float fast_copysign(float value, float sign) noexcept {
	const __m128 sign_mask = _mm_set_ss(-0.0f);
	return _mm_cvtss_f32(_mm_or_ps(_mm_andnot_ps(sign_mask, _mm_set_ss(value)), _mm_and_ps(sign_mask, _mm_set_ss(sign))));
}
static inline float fast_round(float x) noexcept {
	return _mm_cvtss_f32(_mm_round_ss(_mm_setzero_ps(), _mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

static inline __m256 fast_abs(__m256 value) noexcept {
	return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), value);
}
static inline __m256 fast_sign(__m256 sign) noexcept { //copysign(1.0f, x)
	return _mm256_or_ps(_mm256_set1_ps(1.0f), _mm256_and_ps(_mm256_set1_ps(-0.0f), sign));
}
static inline __m256 fast_copysign(__m256 value, __m256 sign) noexcept {
	const __m256 sign_mask = _mm256_set1_ps(-0.0f);
	return _mm256_or_ps(_mm256_andnot_ps(sign_mask, value), _mm256_and_ps(sign_mask, sign));
}
static inline __m256 fast_round(__m256 x) noexcept {
	return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

float f_xor(float a, float b) {
	return _mm_cvtss_f32(_mm_xor_ps(_mm_set_ss(a), _mm_set_ss(b)));
}
float f_or(float a, float b) {
	return _mm_cvtss_f32(_mm_or_ps(_mm_set_ss(a), _mm_set_ss(b)));
}
float f_and(float a, float b) {
	return _mm_cvtss_f32(_mm_and_ps(_mm_set_ss(a), _mm_set_ss(b)));
}
float f_andnot(float a, float b) {
	return _mm_cvtss_f32(_mm_andnot_ps(_mm_set_ss(a), _mm_set_ss(b)));
}

//https://github.com/blender/cycles/blob/3eaf89c3c0b600eabc3d2fd8361fa17abc8f8992/src/util/math_fast.h
static inline float fast_asin(float x) noexcept {
	float f = fast_abs(x);
	f = (f < 1.0f) ? 1.0f - (1.0f - f) : 1.0f; 	/* Clamp and crush denormals. */
	f = glm::half_pi<float>() - sqrtf(1.0f - f) * (1.5707963267f + f * (-0.213300989f + f * (0.077980478f + f * -0.02164095f)));
	return fast_copysign(f, x);
}
//https://github.com/blender/cycles/blob/3eaf89c3c0b600eabc3d2fd8361fa17abc8f8992/src/util/math_fast.h
static inline float fast_atan2(float y, float x) noexcept {
	const float a = fast_abs(x); const float b = fast_abs(y);
	float lo = std::min(a, b), hi = std::max(a, b);
	float k = hi == 0.0f ? 0.0f : lo / hi; //float k = (b == 0.0f) ? 0.0f : (a == b ? 1.0f : (b > a ? a / b : b / a));
	k = 1.0f - (1.0f - k); /* Crush denormals */
	const float k2 = k * k;
	float r = k * (0.43157974f * k2 + 1.0f) / ((0.05831938f * k2 + 0.76443945f) * k2 + 1.0f);
	if (b > a) r = glm::half_pi<float>() - r;
	if (x < 0.0f) r = glm::pi<float>() - r; //especificamente lo que hacen es testear es sign bit
	return fast_copysign(r, y);
}
//https://github.com/blender/cycles/blob/3eaf89c3c0b600eabc3d2fd8361fa17abc8f8992/src/util/math_fast.h
static inline void fast_sincos(float x, float* const __restrict sine, float* const __restrict cosine) noexcept {
	const float qf = fast_round(x * glm::one_over_pi<float>());
	const __m128 sign_mask = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtps_epi32(_mm_set_ss(qf)), 31)); //(static_cast<int32_t>(qf) & 1) != 0 ? -0.0f : 0.0f
	x += qf * (-0.78515625f * 4);
	x += qf * (-0.00024187564849853515625f * 4);
	x += qf * (-3.7747668102383613586e-08f * 4);
	x += qf * (-1.2816720341285448015e-12f * 4);
	x = glm::half_pi<float>() - (glm::half_pi<float>() - x);  // crush denormals
	float x2 = x * x;
	x = _mm_cvtss_f32(_mm_xor_ps(sign_mask, _mm_set_ss(x))); //	if ((static_cast<int32_t>(qf) & 1) != 0) x = -x;
	float su = 2.6083159809786593541503e-06f;     float cu = -2.71811842367242206819355e-07f;
	su = su * x2 - 0.0001981069071916863322258f;  cu = (cu * x2 + 2.47990446951007470488548e-05f);
	su = su * x2 + 0.00833307858556509017944336f; cu = (cu * x2 - 0.00138888787478208541870117f);
	su = su * x2 - 0.166666597127914428710938f;   cu = (cu * x2 + 0.0416666641831398010253906f);
	su = x2 * (su * x) + x; 	                  cu = (cu * x2 - 0.5f); cu = (cu * x2 + 1.0f);
	cu = _mm_cvtss_f32(_mm_xor_ps(sign_mask, _mm_set_ss(cu))); //	if ((static_cast<int32_t>(qf) & 1) != 0) cu = -cu;
	if (fast_abs(su) > 1.0f) { su = 0.0f; }       if (fast_abs(cu) > 1.0f) { cu = 0.0f; }
	*sine = su;                                   *cosine = cu;
}

/* EXPLORATION INTO FAST APROXIMATION OF SINCOS
Hay un video de SimonDev que podria dar un poco en las tecnicas usadas arriba particularmente esa primera polinomial
Es normal que haya dos, en mi metodo aqui habria que evaluar 2 p(x) para sincos y es normal, para un angulo x la posicion dentro del cuadrante no coincide para sin y cos
luego minimo la curva del cuadrante dos veces y luego manejar los cuadrantes 
La primera polynomial con qf lo mismo esta haciendo algo equivalente al branching del parametro de p(x) mapeando el valor en todo el intervalo de una manera uniforme a las curvas del cuadrante


f(x) = {0 < x < 1:sin(x * 2pi)}
p(x) = {0 < x < 0.25 : x * ((4- 16x) * sqrt(x) + (6 - 8x))} //quadrant that we aproximate with a polynomial
sin(x) = {
	0.00 < x < 0.25 : p(x)
	0.25 < x < 0.50 : p(0.5 - x)
	0.50 < x < 0.75 : -p(x - 0.5)
	0.75 < x < 1.00 : -p(1 - x)
}

cos(x) = {
	0.00 < x < 0.25 : p(0.25 - x)
	0.25 < x < 0.50 : -p(x - 0.25)
	0.50 < x < 0.75 : -p(0.75 - x)
	0.75 < x < 1.00 : p(x - 0.75)
}
*/

//computes aproximate of 0.5 - sin^-1(x)/pi
static inline __m256 fast_asin_01(__m256 x) noexcept {
	Vec8f a = fast_abs(x);
#if true
	Vec8f r = 0.5f - sqrt(1.0f - a) * (0.5f + a * (-0.0678958f + (0.024822f - 0.00688853f * a) * a));
	return 0.5f - ((Vec8f{-0.0f} & Vec8f{x}) | r); //fast_copysign(r, x); r > 0
#else
	return 0.5f - 0.5f * ((Vec8f{-0.0f} & Vec8f{x}) | (1.0f - sqrt(1.0f - a)));
#endif
}
//computes aproximate of 0.5 + atan2(y, x)/(2pi)
static inline __m256 fast_atan2_01(__m256 y, __m256 x) noexcept {
	Vec8f a = fast_abs(x), b = fast_abs(y);
	Vec8f lo = min(a, b), hi = max(a, b);
	Vec8f k = _mm256_andnot_ps(hi == 0.0f, lo / hi);
#if true
	Vec8f k2 = k * k;
	Vec8f r = k * (0.43157974f * k2 + 1.0f) / ((0.36643147153982f * k2 + 4.80311472046844413764820f) * k2 + 6.2831853071795864769252f);
#else
	Vec8f r = k * (0.171075f - 0.046075f * k);
#endif
	r = _mm256_blendv_ps(r, 0.25f - r, b > a);
	r = _mm256_blendv_ps(r, 0.5f - r, x);
	return Vec8f{0.5f} + ((Vec8f{-0.0f} & Vec8f{y}) | r); //fast_copysign(r, y); r > 0
}
