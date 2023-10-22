#pragma once

#include <cmath>

#define GLM_FORCE_SIZE_T_LENGTH
#define GLM_FORCE_EXPLICIT_CTOR
#define GLM_FORCE_SWIZZLE
#define GLM_FORCE_INLINE
#define GLM_FORCE_INTRINSICS
#define GLM_FORCE_SIMD_AVX2
#define GLM_FORCE_PRECISION_MEDIUMP_FLOAT

#include <glm/vec2.hpp>               // vec2, bvec2, dvec2, ivec2 and uvec2
#include <glm/vec3.hpp>               // vec3, bvec3, dvec3, ivec3 and uvec3
#include <glm/vec4.hpp>               // vec4, bvec4, dvec4, ivec4 and uvec4
#include <glm/mat4x4.hpp>             // mat4, dmat4
#include <glm/common.hpp>             // all the GLSL common functions: abs, min, mix, isnan, fma, etc.
#include <glm/exponential.hpp>        // all the GLSL exponential functions: pow, log, exp2, sqrt, etc.
#include <glm/geometric.hpp>          // all the GLSL geometry functions: dot, cross, reflect, etc.
#include <glm/matrix.hpp>             // all the GLSL matrix functions: transpose, inverse, etc.
#include <glm/trigonometric.hpp>      // all the GLSL trigonometric functions: radians, cos, asin, etc.
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "vectorclass.h"
#include "vectormath_exp.h"
#include "vectormath_trig.h"

#define INLINE [[msvc::forceinline]] static inline
#define VECTORCALL __vectorcall

#include "iacaMarks.h"

using Spectrum = glm::vec3;