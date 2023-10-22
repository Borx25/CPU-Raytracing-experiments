#pragma once
#include <array>
#include "Bitmanip.hpp"

static inline constexpr float make_unit_float(uint32_t x) noexcept { return static_cast<float>(x) * 0x1p-32f; }

//este es el que usa cherno excepto que usa el return value como siguiente state en vez de pasarle solo el lcg step
//https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/

static inline constexpr uint32_t pcg_state_transition(uint32_t val) noexcept
{
    return val * 747796405u + 2891336453u;
}
static inline constexpr uint32_t pcg_output(uint32_t val) noexcept
{
   val = ((val >> ((val >> 28u) + 4u)) ^ val) * 277803737u;
    return (val >> 22u) ^ val;
}

static inline constexpr uint32_t pcg_generate(uint32_t* state) noexcept {
    uint32_t prev_state = *state;
    *state = pcg_state_transition(prev_state);
    return pcg_output(prev_state);
}

static inline float rand_unit_float(uint32_t* state)
{
    return make_unit_float(pcg_generate(state));
}

static inline uint32_t rand_bounded_int(uint32_t* state, uint32_t range) noexcept
{
    return std::min(range - 1, static_cast<uint32_t>(rand_unit_float(state) * static_cast<float>(range)));
}

static inline constexpr uint32_t hash_u32(uint32_t i) noexcept { // From https://github.com/skeeto/hash-prospector
    i ^= i >> 16;
    i *= 0x21f0aaad;
    i ^= i >> 15;
    i *= 0xd35a2d97;
    i ^= i >> 15;
    return i ^ 0xe6fe3beb; //make input zero not map to output zero. number is randomly selected and isn't special.
}

static inline constexpr uint32_t hash_2d(const uint32_t x, const uint32_t y) noexcept
{
    const uint32_t qx = 0x41c64e6du * ((x >> 1U) ^ (y));
    const uint32_t qy = 0x41c64e6du * ((y >> 1U) ^ (x));
    return 0x41c64e6du * (qx ^ (qy >> 3U));
}