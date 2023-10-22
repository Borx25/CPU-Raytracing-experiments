#pragma once
#include <stdint.h>
#include <intrin.h>
#include <type_traits> //std::is_constant_evaluated
#include <bit>


inline constexpr int32_t float_exponent(float x) noexcept {
    return ((std::bit_cast<int32_t>(x) & 0x7f800000) >> 23) - 127;
}

template<std::integral T> inline constexpr bool is_power_of_two(T x) {
    return (x > 0) && !(x & (x - 1));
}

template<std::integral T>
static inline constexpr T round_up_pow2(T val) noexcept {
    --val;
    for (size_t shift = 1; shift < sizeof(T) * CHAR_BIT; shift <<= 1) {
        val |= val >> shift;
    }
    return ++val;
}

template<std::integral T>
static inline constexpr T trailingzeros(T val) noexcept {
#ifdef __AVX2__ 
    if (!std::is_constant_evaluated()) {
        if constexpr (sizeof(T) <= 4) {
            return static_cast<T>(_tzcnt_u32(static_cast<uint32_t>(val)));
        } else {
            return static_cast<T>(_tzcnt_u64(static_cast<uint64_t>(val)));
        }
    }
#endif
    if (val == 0) return  CHAR_BIT * sizeof(T);
    T bit = 0;
    val = (val ^ (val - 1)) >> 1;  // Set v's trailing 0s to 1s and zero rest
    for (bit = 0; val; bit++) val >>= 1;
    return bit;
}

template<std::integral T>
static inline constexpr T leadingzeros(T val) noexcept {
#ifdef __AVX2__ 
    if (!std::is_constant_evaluated()) {
        if constexpr (sizeof(T) <= 4) {
            return static_cast<T>(_lzcnt_u32(static_cast<uint32_t>(val)));
        } else {
            return static_cast<T>(_lzcnt_u64(static_cast<uint64_t>(val)));
        }
    }
#endif
    return static_cast<T>(sizeof(T) * CHAR_BIT - trailingzeros(round_up_pow2(val)));
}


template<std::integral T>
static inline constexpr T reset_lsb(T val) noexcept {
#ifdef __AVX2__ 
    if (!std::is_constant_evaluated()) {
        if constexpr (sizeof(T) <= 4) {
            return static_cast<T>(_blsr_u32(static_cast<uint32_t>(val)));
        } else {
            return static_cast<T>(_blsr_u64(static_cast<uint64_t>(val)));
        }
    }
#endif
    return val & (val - 1);
}

template<std::integral T>
static inline constexpr T isolate_lsb(const T val) noexcept {
#ifdef __AVX2__ 
    if (!std::is_constant_evaluated()) {
        if constexpr (sizeof(T) <= 4) {
            return static_cast<T>(_blsi_u32(static_cast<uint32_t>(val)));
        } else {
            return static_cast<T>(_blsi_u64(static_cast<uint32_t>(val)));
        }
    }
#endif
    return val & static_cast<T>(-static_cast<std::make_signed_t<T>>(val));
}

template<std::integral  T>
static inline constexpr T deposit_bits(T val, T mask) noexcept {
#ifdef __AVX2__ 
    if (!std::is_constant_evaluated()) {
        if constexpr (sizeof(T) <= 4) {
            return static_cast<T>(_pdep_u32(static_cast<uint32_t>(val), static_cast<uint32_t>(mask)));
        } else {
            return static_cast<T>(_pdep_u64(static_cast<uint64_t>(val), static_cast<uint64_t>(mask)));
        }
    }
#endif
    T accum = static_cast<T>(0);
    for (T bit_mask = static_cast<T>(1); !!mask; bit_mask += bit_mask) {
        if (!!(val & bit_mask)) accum |= isolate_lsb(mask);
        mask = reset_lsb(mask);
    }
    return accum;
}

template<std::integral T>
static inline constexpr T gather_bits(const T val, T mask) noexcept {
#ifdef __AVX2__ 
    if (!std::is_constant_evaluated()) {
        if constexpr (sizeof(T) <= 4) {
            return static_cast<T>(_pext_u32(static_cast<uint32_t>(val), static_cast<uint32_t>(mask)));
        } else {
            return static_cast<T>(_pext_u64(static_cast<uint64_t>(val), static_cast<uint64_t>(mask)));
        }
    }
#endif
    T accum = static_cast<T>(0);
    for (T bit_mask = static_cast<T>(1); !!mask; bit_mask += bit_mask) {
        if (!!(val & isolate_lsb(mask))) accum |= bit_mask;
        mask = reset_lsb(mask);
    }
    return accum;
}

template<std::integral T>
static inline constexpr T popcount(T val) noexcept {
#ifdef __AVX2__ 
    if (!std::is_constant_evaluated()) {
        if constexpr (sizeof(T) <= 2) {
            return static_cast<T>(__popcnt16(static_cast<uint16_t>(val)));
        } else if constexpr (sizeof(T) <= 4) {
            return static_cast<T>(__popcnt(static_cast<uint32_t>(val)));
        } else {
            return static_cast<T>(__popcnt64(static_cast<uint64_t>(val)));
        }
    }
#endif
    constexpr auto bit_patterns = []<size_t... i>(std::index_sequence<i...>) consteval {
        return std::array{static_cast<T>(~0ull / (1ull << (1 << i) | 1ull))..., static_cast<T>(~0ull / 0xFF)};
    }(std::make_index_sequence<3>{});

    uint64_t accum = static_cast<uint64_t>(val);
    accum -= (accum >> 1) & bit_patterns[0];
    accum = (accum & bit_patterns[1]) + ((accum >> 2) & bit_patterns[1]);
    accum = (accum + (accum >> 4)) & bit_patterns[2];
    return static_cast<T>(static_cast<uint64_t>(accum * bit_patterns[3]) >> ((sizeof(T) - 1) * CHAR_BIT));
}


template<std::integral T>
static inline constexpr T permute_bits(const T val, std::type_identity_t<T> mask, const unsigned shift) noexcept {
    return ((val & mask) << shift) | ((val >> shift) & mask);
}

template<std::integral W, std::integral T> requires(sizeof(T) > sizeof(W))
static inline constexpr T word_swap(T val) noexcept {
#ifdef __AVX2__ 
    if constexpr (sizeof(W) == 1) {
        if (!std::is_constant_evaluated()) {
            if constexpr (sizeof(T) == 2) {
                return static_cast<T>(_byteswap_ushort(static_cast<uint16_t>(val)));
            } else if constexpr (sizeof(T) == 4) {
                return static_cast<T>(_byteswap_ulong(static_cast<uint32_t>(val)));
            } else {
                return static_cast<T>(_byteswap_uint64(static_cast<uint64_t>(val)));
            }
        }
    }
#endif
    constexpr size_t N = (sizeof(T) / sizeof(W)) / 2;
    constexpr size_t width = CHAR_BIT * sizeof(W);

    return [] <auto... i>(T val, std::index_sequence<i...>) {
        return (... | permute_bits(val, static_cast<T>(static_cast<W>(-1)) << (width * (N - (i + 1))), static_cast<int>(width * (i * 2 + 1))));
    }(val, std::make_index_sequence<N>{});
}

template<std::integral T> requires(sizeof(T) > 1) static inline constexpr T byteswap(T val) noexcept { return word_swap<uint8_t>(val); }

static constexpr uint8_t reversed_nibbles[16]{
    //0 => 0000 => 0000 => 0
    //1 => 0001 => 1000 => 8
    //2 => 0010 => 0100 => 4
    //3 => 0011 => 1100 => C
    //4 => 0100 => 0010 => 2
    //5 => 0101 => 1010 => A
    //6 => 0110 => 0110 => 6
    //7 => 0111 => 1110 => E
    //8 => 1000 => 0001 => 1
    //9 => 1001 => 1001 => 9
    //A => 1010 => 0101 => 5
    //B => 1011 => 1101 => D
    //C => 1100 => 0011 => 3
    //D => 1101 => 1011 => B
    //E => 1110 => 0111 => 7
    //F => 1111 => 1111 => F
    0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE, 0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF
};


template<std::integral T> static inline constexpr T bitreverse(T val) noexcept {
    //instrinsics when available, there are for cuda for example: 
    //https://github.com/blender/cycles/blob/3eaf89c3c0b600eabc3d2fd8361fa17abc8f8992/src/util/math.h#L967
    if constexpr (sizeof(T) == 1) {
        return static_cast<T>((reversed_nibbles[0x0F & static_cast<uint8_t>(val)] << 4) | (reversed_nibbles[static_cast<uint8_t>(val) >> 4]));
    } else {
    //#ifdef __AVX2__ 
    //    if (!std::is_constant_evaluated()) {
    //        __m128i temp;
    //        if constexpr (sizeof(T) <= 4) {
    //            temp = _mm_cvtsi32_si128(static_cast<int32_t>(val));
    //        } else {
    //            temp = _mm_cvtsi64_si128(static_cast<int64_t>(val));
    //        }
    //        const __m128i LUT = _mm_load_si128((__m128i*) & reversed_nibbles);
    //        const __m128i low_nibble = _mm_set1_epi8(0x0F);
    //        temp = _mm_or_si128(
    //            _mm_shuffle_epi8(_mm_slli_epi32(LUT, 4), _mm_and_si128(low_nibble, temp)),
    //            _mm_shuffle_epi8(LUT, _mm_and_si128(low_nibble, _mm_srli_epi32(temp, 4)))
    //        );
    //        if constexpr (sizeof(T) <= 4) {
    //            val = static_cast<T>(_mm_cvtsi128_si32(temp));
    //        } else {
    //            val = static_cast<T>(_mm_cvtsi128_si64(temp));
    //        }
    //        return byteswap(val);
    //    }
    //#endif
        val = permute_bits(val, static_cast<T>(0x5555555555555555), 1);
        val = permute_bits(val, static_cast<T>(0x3333333333333333), 2);
        val = permute_bits(val, static_cast<T>(0x0F0F0F0F0F0F0F0F), 4);
        return byteswap(val);
    }
}

template<std::integral T>
static inline constexpr T morton_encode2D(T x, T y) noexcept {
    return deposit_bits<T>(x, static_cast<T>(0x5555555555555555ull)) | deposit_bits<T>(y, static_cast<T>(0xAAAAAAAAAAAAAAAAull));
}

template<std::integral T> struct morton_decode_ret { T x, y; };
template<std::integral T> static inline constexpr morton_decode_ret<T> morton_decode2D(T val) noexcept {
    return {
        gather_bits<T>(val, 0x5555555555555555ull),
        gather_bits<T>(val, 0xAAAAAAAAAAAAAAAAull)
    };
}

//alternativamente a pdep para el caso de morton code se puede hacer lo siguiente:
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
//unsigned int expandBits(unsigned int v)
//{
//    v = (v * 0x00010001u) & 0xFF0000FFu;
//    v = (v * 0x00000101u) & 0x0F00F00Fu;
//    v = (v * 0x00000011u) & 0xC30C30C3u;
//    v = (v * 0x00000005u) & 0x49249249u;
//    return v;
//}