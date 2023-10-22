#pragma once

#include "Sampling.hpp"
#include "Bitmanip.hpp"

template<size_t bits>
struct bitset {
	static constexpr size_t count = std::min(CHAR_BIT * sizeof(uint64_t), std::max(static_cast<size_t>(CHAR_BIT), round_up_pow2(bits)));
	static constexpr size_t N = (bits + count - 1) / count;
	using T = UnsignedIntType<count>;
	T block[N];

	bool test(size_t i) const {
		return (block[i / count] & (static_cast<T>(1ull) << (i % count))) != 0;
	}
	void set(size_t i) {
		block[i / count] |= (static_cast<T>(1ull) << (i % count));
	}
	void reset(size_t i) {
		block[i / count] &= ~(static_cast<T>(1ull) << (i % count));
	}
	void flip(size_t i) {
		block[i / count] ^= (static_cast<T>(1ull) << (i % count));
	}
	void set(size_t i, bool value) {
		block[i / count] ^= (static_cast<T>(-static_cast<SignedIntType<count>>(value)) ^ block[i / count]) & (static_cast<T>(1ull) << (i % count));
	}
	void bit_or(size_t i, bool value) {
		block[i / count] |= (static_cast<T>(value) << (i % count));
	}
	void bit_and(size_t i, bool value) {
		block[i / count] &= (static_cast<T>(value) << (i % count));
	}
	void bit_xor(size_t i, bool value) {
		block[i / count] ^= (static_cast<T>(value) << (i % count));
	}
	void bit_nand(size_t i, bool value) {
		block[i / count] &= ~(static_cast<T>(value) << (i % count));
	}
	void zero() {
		memset(block, 0, sizeof(T) * N);
	}
	void ones() {
		memset(block, ~0, sizeof(T) * N);
	}
	void set_all(int byte = ~0) {
		memset(block, byte, sizeof(T) * N);
	}

	size_t popcount() const {
		size_t ret = 0;
		for (size_t i = 0; i < N; i++) {
			ret += ::popcount<T>(block[N]);
		}
		return ret;
	}
};

struct bitscan_sentinel {};
template<typename T>
struct bitscan_iterator {
	T val;
	constexpr T operator*() const noexcept {
		return trailingzeros<T>(val);
	}
	constexpr bitscan_iterator& operator++() noexcept {
		val = reset_lsb<T>(val);
		return *this;
	}
	constexpr bool operator==(const bitscan_sentinel& s) const noexcept { return !val; };
};


template<size_t Size> struct alignas(64) RayStream {
	struct Buffer {
		struct {
			float x[Size];
			float y[Size];
			float z[Size];
		} p, dir;
		struct {
			float r[Size];
			float g[Size];
			float b[Size];
		} radiance, throughput;
		float pdf[Size];
		uint32_t pixelID[Size]; //Pixel ID (within the tile so 0 to N), could pack more stuff or be a smaller type
	};
	struct {
		bitset<Size> termination;
		bitset<Size> has_shadowray;
	} flags;
	uint32_t seed[Size];
	uint32_t RayID[Size];
	struct Path {
		Path() : input(&buffers[0]), output(&buffers[1]) {}
		Buffer* const __restrict input;
		Buffer* const __restrict output;
		void swap() noexcept {
			auto tmp = const_cast<Buffer*>(input);
			const_cast<Buffer*&>(input) = output;
			const_cast<Buffer*&>(output) = tmp;
		}
		Buffer buffers[2];
	} path;
	struct Hit {
		float tfar[Size];
		int32_t primID[Size];
		int32_t matID[Size]; //doesn't need entire range, could pack additional related material, brdfs o lo q sea
		//float u[Size];
		//float v[Size];
	} hit;
	struct ShadowStream {
		struct {
			float x[Size];
			float y[Size];
			float z[Size];
		} p, dir;
		float tfar[Size];
		struct {
			float r[Size];
			float g[Size];
			float b[Size];
		} radiance;
		bitset<Size> occluded;
	} shadow_rays;

};

enum class ClosureType : int32_t {
	LambertianDiffuse,
	GGX
};

struct alignas(64) UndefinedClosure {
	ClosureType type;
	char storage[60];
};

static_assert(sizeof(UndefinedClosure) == 64);

template<size_t Size> struct alignas(64) ShaderDataStream {
	UndefinedClosure closure[Size];
	struct {
		float x[Size];
		float y[Size];
		float z[Size];
	} P, V;
	struct {
		float x[Size];
		float y[Size];
		float z[Size];
		float w[Size];
	} T;
	float light_sample_prob[Size];
	bitset<Size> is_emissive;
};
template<ClosureType> struct alignas(64) Closure;

struct Sample {
	glm::vec3 dir;
	Spectrum estimator;
};

template<> struct alignas(64) Closure<ClosureType::LambertianDiffuse> {
	ClosureType type;
	Spectrum albedo;

	Spectrum eval(glm::vec3 Llocal, glm::vec3 Vlocal) noexcept {
		float NdotL = std::max(0.0f, Llocal.z);
		return albedo * (glm::one_over_pi<float>() * NdotL);
	}
	float pdf(glm::vec3 Llocal) noexcept {
		float NdotL = std::max(0.0f, Llocal.z);
		return glm::one_over_pi<float>() * NdotL;
	}
	bool sample(Sample& sample, glm::vec3 Vlocal, const float* const __restrict random) noexcept {
		sample.dir = hemisphere(random[0], random[1]); //hemisphere weighted with pdf = NdotL / PI
		sample.estimator = albedo;
		return true;
	}
};

template<> struct alignas(64) Closure<ClosureType::GGX> {
	ClosureType type;
	Spectrum F0;
	float alpha;

	Spectrum eval(glm::vec3 Llocal, glm::vec3 Vlocal) noexcept {
		float NdotL = std::max(0.0f, Llocal.z);
		float NdotV = std::max(0.0f, Vlocal.z);
		float NdotH = std::max(0.0f, glm::normalize(Llocal + Vlocal).z);
		float HdotV = std::max(0.0f, glm::dot(glm::normalize(Llocal + Vlocal), Vlocal));
		return microfacet_brdf(F0, alpha, NdotV, NdotL, NdotH, HdotV);
	}
	float pdf(glm::vec3 Llocal) noexcept {
		return 0.0f; //TODO
	}
	bool sample(Sample& sample, glm::vec3 Vlocal, const float* const __restrict random) noexcept {
		float NdotV = std::max(0.0f, Vlocal.z);
		float HdotV;
		if (alpha == 0.0f) {
			sample.dir = glm::vec3{
				-Vlocal.x,
				-Vlocal.y,
				Vlocal.z
			};
			HdotV = NdotV;
		} else {
			glm::vec3 Hlocal = distribution_visible_normals(Vlocal, alpha, random[0], random[1]);
			HdotV = glm::dot(Hlocal, Vlocal);
			sample.dir = (2.0f * HdotV) * Hlocal - Vlocal;
			HdotV = std::max(0.0f, HdotV);
		}
		float NdotL = std::max(0.0f, sample.dir.z);
		sample.estimator = vndf_estimator(F0, alpha, NdotV, NdotL, HdotV);
		return true;
	}
};

template<typename T, typename U> static inline void histogram(const T* const __restrict in, U* const __restrict out, uint32_t n) {
	const T* ptr = in;
	const T* const end = in + n;
	const T* const unroll_end = in + n - 3;
	for (; ptr < unroll_end; ptr += 4) {
		++out[ptr[0]];
		++out[ptr[1]];
		++out[ptr[2]];
		++out[ptr[3]];
	}
	for (; ptr < end; ptr++) ++out[*ptr];
}
template<typename T> static inline void prefix_sum(T* const __restrict ptr, uint32_t n) {
	for (uint32_t i = 1; i < n; i++) ptr[i] += ptr[i - 1]; //n will typically be very small, not worth doing simd
}
static inline void counting_sort(const int32_t* const __restrict key, uint16_t* const __restrict histogram_prefix_sum, uint32_t* const __restrict out, uint32_t n) {
	for (int32_t i = n - 1; i >= 0; i--) out[--histogram_prefix_sum[key[i]]] = i;
}
template<size_t upper_k = 0xFF, size_t upper_count = 0xFFFF>
static inline size_t sort_rayID(
	const uint32_t k, const uint32_t count, 
	uint32_t* const __restrict out, 
	const int32_t* const __restrict key_ptr, 
	uint16_t* const __restrict sort_buffer
) noexcept {
	__assume(0 < k && k <= upper_k);
	__assume(0 < count && count <= upper_count);
	histogram(key_ptr, sort_buffer + 1, count);//instead of out[in[i] + 1] inside just advance out pointer
	size_t ret = sort_buffer[0]; //save miss count
	prefix_sum(sort_buffer, k + 1); //items with a particular key should be placed starting at the corresponding position in the buffer
	counting_sort(key_ptr, sort_buffer + 1, out, count);
	return ret;
};