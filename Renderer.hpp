#pragma once
#include <vector>
#include <algorithm>
#include <execution>
#include <bit>
#include <ranges>
#include <stack>
#include <ppl.h>

#include "Image.h"

#include "Core.hpp"
#include "VectorMath.hpp"
#include "Scene.hpp"
#include "Color.hpp"
#include "Sampling.hpp"


struct RendererPolicy {
	size_t log_tile = 4; //Tiles of (2^log_tile)^2 pixels
	size_t samples_per_pixel = 1; //por ahora que esto sea mas de uno no esta implementado
	//la unica motivacion es si hubiera beneficio de tener un stream mas grande sin aumentar el tile que el paper de streams si que recomendaba streams de 2048 rays siempre con tiles de 16x16
	size_t max_materialID = 64;
	size_t max_bounces = 16;
	float max_radiance = 1e2f;
};

template<RendererPolicy _Policy = RendererPolicy{}>
struct Renderer {
	static constexpr RendererPolicy Policy = _Policy;
	static_assert(Policy.log_tile >= 3, "_Policy.log_tile must be at least 3 due to avx2");
	static constexpr size_t TileRoot = 1ull << Policy.log_tile;
	static constexpr size_t TileSize = TileRoot * TileRoot;
	static constexpr size_t StreamSize = TileSize * Policy.samples_per_pixel;

	static constexpr size_t RequiredTiling() { return TileRoot; }

	const Scene& scene;
	std::unique_ptr<Image> output_frame;
	std::vector<glm::vec4> framebuffer;
	static constexpr size_t AccumulationBuckets = 5;

	template<size_t k> requires ((k & 1) != 0)
	struct AccumulationTile {
		struct { float r[TileSize]; float g[TileSize]; float b[TileSize]; } color[k];
	}; std::vector<AccumulationTile<AccumulationBuckets>> accumulator;
	uint32_t width = 0, height = 0;
	uint32_t accumulations = 0;
	uint32_t h_tiles = 0, v_tiles = 0;

	Renderer(const Scene& scene) : scene(scene) {}

	void Resize(uint32_t new_width, uint32_t new_height) {
		height = new_height;
		width = new_width;
		if (!output_frame) output_frame = std::make_unique<Image>(width, height, ImageFormat::RGBA32F);
		output_frame->Resize(width, height);
		framebuffer.resize(static_cast<size_t>(width * height));
		h_tiles = width / TileRoot;
		v_tiles = height / TileRoot;
		accumulator.resize(static_cast<size_t>(h_tiles * v_tiles));
		ResetAccumulator();
	}
	void ResetAccumulator() {
		accumulations = 0;
		memset(accumulator.data(), 0, accumulator.size() * sizeof(AccumulationTile<AccumulationBuckets>));
	}
	auto& GetFrame() { return output_frame; }

#define BRDF 0
#define MIS true

	void Accumulate() {
		++accumulations;
		concurrency::parallel_for(0u, static_cast<uint32_t>(accumulator.size()), [
			this,
			light_count = static_cast<uint32_t>(scene.lighting_acceleration.prims.size()),
			light_selection_pdf = 1.0f / static_cast<float>(scene.lighting_acceleration.prims.size()),
			has_ambient = std::max(scene.sky.ambient_color.r, std::max(scene.sky.ambient_color.g, scene.sky.ambient_color.b)) > 0.0f,
			sample_index = bitreverse(accumulations),
			frame_seed = hash_u32(accumulations),
			bucket_index = accumulations % AccumulationBuckets
		] (const uint32_t LaunchIndex) {
			auto& output_color = accumulator[LaunchIndex].color[bucket_index];
			const struct { int32_t x, y; } tile {
				 TileRoot * (LaunchIndex % h_tiles),
				 TileRoot * (LaunchIndex / h_tiles)
			};
			//Podria mergear todos en en solo struct del state per tile
			RayStream<StreamSize> ray_stream;
			ShaderDataStream<StreamSize> shaderdata_stream;
			uint16_t sort_buffer[Policy.max_materialID/*scene.material.size() + 1*/]; //buffer for counting sort
			auto* const __restrict initial_buffer = const_cast<decltype(ray_stream)::Buffer*>(&ray_stream.path.buffers[0]);
			/*---------------------------------
			----------- INITIAL SETUP ---------
			---------------------------------*/
			{
				auto* const __restrict radiance_ptr = reinterpret_cast<float*>(&initial_buffer->radiance);
				for (size_t i = 0; i < StreamSize / 8 * 3; i++) _mm256_store_ps(radiance_ptr + i * 8, _mm256_setzero_ps());
				auto* const __restrict throughput_ptr = reinterpret_cast<float*>(&initial_buffer->throughput);
				for (size_t i = 0; i < StreamSize / 8 * 3; i++) _mm256_store_ps(throughput_ptr + i * 8, _mm256_set1_ps(1.0f));
				__m256i pixelID = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
				auto* const __restrict pixelID_ptr = reinterpret_cast<__m256i*>(&initial_buffer->pixelID);
				for (size_t i = 0; i < StreamSize / 8; i++, pixelID = _mm256_add_epi32(pixelID, _mm256_set1_epi32(8))) _mm256_store_si256(pixelID_ptr + i, pixelID);
				//SEED
				for (size_t ID = 0; ID < TileSize; ID++) {
					ray_stream.seed[ID] = static_cast<int32_t>((LaunchIndex * TileSize + ID) * (Policy.max_bounces * 2 + 1)); //2 branches of the lcg per bounce (brdf and light sampling) + 1 for camera
				}
			}
			/*---------------------------------
			---------- RAY GENERATION ---------  //Profiling: 12%
			---------------------------------*/
			for (size_t ID = 0; ID < TileSize; ID++) {
				int32_t x = tile.x + static_cast<int32_t>(ID) % TileRoot;
				int32_t y = tile.y + static_cast<int32_t>(ID) / TileRoot;

				uint32_t rng_state = hash_2d(accumulations, ray_stream.seed[ID]);
				const std::array<float, 2> camera_samples{rand_unit_float(&rng_state), rand_unit_float(&rng_state)};

				auto [orig, dir] = scene.camera.generate_ray(x, y, camera_samples.data());
				initial_buffer->dir.x[ID] = dir .x;
				initial_buffer->dir.y[ID] = dir .y;
				initial_buffer->dir.z[ID] = dir .z;
				initial_buffer->p  .x[ID] = orig.x;
				initial_buffer->p  .y[ID] = orig.y;
				initial_buffer->p  .z[ID] = orig.z;
			}
			/*---------------------------------
			----------- BOUNCE LOOP -----------
			---------------------------------*/
			for (size_t bounce = 0, active_rays = StreamSize; bounce < Policy.max_bounces && active_rays > 0; bounce++, ray_stream.path.swap()) {
				auto* const __restrict in = ray_stream.path.input;
				auto* const __restrict out = ray_stream.path.output;

				{//Zero flags
					ray_stream.flags.termination.zero();
					ray_stream.flags.has_shadowray.zero();
					ray_stream.shadow_rays.occluded.zero();
					shaderdata_stream.is_emissive.zero();
				}
				{//Zero sort buffer
					auto* sort_buffer_ptr = reinterpret_cast<__m256i*>(&sort_buffer);
				#if false
					const auto* sort_buffer_end = sort_buffer_ptr + (scene.material.size() + 1 + 15) / 16;
					for (; sort_buffer_ptr < sort_buffer_end; ++sort_buffer_ptr) _mm256_store_si256(sort_buffer_ptr, _mm256_setzero_si256());
				#else //sort buffer is very small, save the overhead of working out how much needs to be zeroed
					for (size_t i = 0; i < Policy.max_materialID / 16; i++) _mm256_store_si256(sort_buffer_ptr + i, _mm256_setzero_si256());
				#endif
				}
				{//Reset hit struct
					auto* const __restrict tfar_ptr = reinterpret_cast<float*>(&ray_stream.hit.tfar);
					auto* const __restrict matID_ptr = reinterpret_cast<__m256i*>(&ray_stream.hit.matID);
					auto* const __restrict primID_ptr = reinterpret_cast<__m256i*>(&ray_stream.hit.primID);
					for (size_t i = 0; i < (active_rays + 7) / 8; i++) {
						_mm256_store_ps(tfar_ptr + i * 8, _mm256_set1_ps(FLT_MAX)); //se podrian resetear enteros independientemente de active
						_mm256_store_si256(matID_ptr + i, _mm256_set1_epi32(-1)); //este ojo si sort_rayID es ciego y ordena todo, entonces hace falta marcar con un maximo pasado active q este loop ahora se pasa
						_mm256_store_si256(primID_ptr + i, _mm256_set1_epi32(-1)); //este tmb
					}
					//for (size_t i = active_rays; i < StreamSize; i++) //mark as dead, not actually needed if sorting stops at active_rays
					//	ray_stream.hit.matID[i] = 254; //so their key will be 255 and be positioned last
				}
				/*---------------------------------
				----------- INTERSECTION ---------- //Profiling: 10%
				---------------------------------*/
				scene.acceleration_structure.Traverse<StreamSize>(*in, ray_stream.hit, active_rays);
				/*---------------------------------
				------  CLOSEST HIT SHADER ------- //Profiling: 10%
				---------------------------------*/
				for (size_t ID = 0; ID < active_rays; ID++) {
					const int32_t& mat_ID = ray_stream.hit.matID[ID];
					if (mat_ID == -1) continue;
					const int32_t& prim_ID = ray_stream.hit.primID[ID];
					const float& depth = ray_stream.hit.tfar[ID];
					const glm::vec3 D{ in->dir.x[ID], in->dir.y[ID], in->dir.z[ID] };

					glm::vec3 hit_point{
						in->p.x[ID] + D.x * depth,
						in->p.y[ID] + D.y * depth,
						in->p.z[ID] + D.z * depth
					};
					glm::vec3 N{
						hit_point.x - scene.acceleration_structure.prims[prim_ID].position.x,
						hit_point.y - scene.acceleration_structure.prims[prim_ID].position.y,
						hit_point.z - scene.acceleration_structure.prims[prim_ID].position.z
					};
					N = glm::normalize(N);
					if (glm::dot(N, D) >= 0.0f) N = -N; //backface
					glm::quat T = tangent_space(N);
					glm::vec3 Vlocal = to_local(T, -D);
					shaderdata_stream.P.x[ID] = hit_point.x + N.x * 1e-4f;
					shaderdata_stream.P.y[ID] = hit_point.y + N.y * 1e-4f;
					shaderdata_stream.P.z[ID] = hit_point.z + N.z * 1e-4f;
					shaderdata_stream.V.x[ID] = Vlocal.x;
					shaderdata_stream.V.y[ID] = Vlocal.y;
					shaderdata_stream.V.z[ID] = Vlocal.z;
					shaderdata_stream.T.x[ID] = T.x;
					shaderdata_stream.T.y[ID] = T.y;
					shaderdata_stream.T.z[ID] = T.z;
					shaderdata_stream.T.w[ID] = T.w;
					//emission flag
					if (std::max(scene.material[mat_ID].emission.r, std::max(scene.material[mat_ID].emission.g, scene.material[mat_ID].emission.b)) > FLT_EPSILON) {
						shaderdata_stream.is_emissive.set(ID);
					}
					//---- closure ----
					auto* const __restrict closure_ptr = reinterpret_cast<Closure<static_cast<ClosureType>(BRDF)>*>(&shaderdata_stream.closure[ID]);
					closure_ptr->type = static_cast<ClosureType>(BRDF);
				#if BRDF == 0
					closure_ptr->albedo = scene.material[mat_ID].albedo;
				#else
					closure_ptr->F0 = scene.material[mat_ID].F0;
					float alpha = scene.material[mat_ID].roughness; alpha *= alpha;
					closure_ptr->alpha = alpha + (1.0f - alpha) * gloss_decay_table[bounce]; // Reduce gloss on later bounces
				#endif
				}
				/*---------------------------------
				------- FIRST BOUNCE OUTPUTS ------
				---------------------------------*/
			#if false
				if (bounce == 0) {
					for (size_t i = 0; i < miss_count; i++) {
						accumulation_tile.depth[ray_stream.RayID[i]] += 1e4;
					}
					for (size_t i = 0; i < hit_count; i++) {
						const int32_t& ID = ray_stream.RayID[miss_count + i];
						accumulation_tile.depth[ID] += ray_stream.hit.tfar[ID];
						accumulation_tile.normal.x[ID] += shaderdata_stream.N.x[i];
						accumulation_tile.normal.y[ID] += shaderdata_stream.N.y[i];
						accumulation_tile.normal.z[ID] += shaderdata_stream.N.z[i];
					}
				}
			#endif
				/*---------------------------------
				------- COHERENCE EXTRACTION ------ //Profiling: 4%
				---------------------------------*/
				const size_t miss_count = sort_rayID<Policy.max_materialID, StreamSize>(
					scene.material.size(),
					active_rays,
					ray_stream.RayID,
					ray_stream.hit.matID,
					sort_buffer
				);

				const size_t hit_count = active_rays - miss_count;
				/*---------------------------------
				------ NEXT EVENT ESTIMATION ------  //Profiling: 27% => sample_direction_to_sphere [8%]
				---------------------------------*/
			#if MIS
				size_t shadow_index = 0;
				for (size_t i = 0; i < hit_count; i++) {
					const int32_t ray_index = miss_count + i;
					const int32_t& ID = ray_stream.RayID[ray_index];
					const uint32_t& seed = ray_stream.seed[in->pixelID[ID]];
					//es necesario la indireccion en seed/rngstate? importa algo que paths se intercambien seeds? es decir pq no ray_stream.seed[i]
					//notese que el paper original no menciona seeds o states de rng como parte del path, dicen q usan sobol con cransley rotation que tengo que mirarlo
					uint32_t rng_state = hash_2d(accumulations, ray_stream.seed[in->pixelID[ID]] + bounce * 2);

					const std::array<float, 2> LightSamples{rand_unit_float(&rng_state), rand_unit_float(&rng_state)};

					int32_t selected_light = static_cast<int32_t>(rand_bounded_int(&rng_state, light_count));
					//COSA A INVESTIGAR: instead of uniform it could pick proportional to the lights power or voxelize the scene and pick proportional to an estimate of its contribution to a particular region
					int32_t light_primID = scene.lighting_acceleration.prims[selected_light];
					const auto& light_prim = scene.geometry[light_primID];
					if (light_primID == ray_stream.hit.primID[ID]) [[unlikely]] continue; //self
					glm::vec3 Wc = light_prim.position - glm::vec3{shaderdata_stream.P.x[ID], shaderdata_stream.P.y[ID], shaderdata_stream.P.z[ID]}; //esto no es del todo correcto pq este P es el offset by normal
					float center_dist2 = glm::dot(Wc, Wc);
					if (center_dist2 <= light_prim.radius_sq) [[unlikely]] continue;// encompasing sphere can be solved efficiently by indirect lighting
					float center_dist = sqrt(center_dist2);
					Wc *= 1.0f / center_dist;
					float sinThetaMax2 = light_prim.radius_sq / center_dist2;
					{//This check is not strictly necessary, its only to exit early, Would be simpler if I had worldspace N available at this point instead of the quaternion
						float NdotW = (2.0f * shaderdata_stream.T.w[ID]) * (Wc.z * shaderdata_stream.T.w[ID] + Wc.x * shaderdata_stream.T.y[ID] - shaderdata_stream.T.x[ID] * Wc.y) - Wc.z; //to_local(T, W).z
						if (NdotW < 0.0f && sinThetaMax2 < NdotW * NdotW) [[unlikely]] continue; //entire cone below hemisphere
					}
					float light_distance, light_pdf; glm::vec3 L = sample_direction_to_sphere(Wc, sinThetaMax2, center_dist, light_prim.radius_sq, LightSamples[0], LightSamples[1], &light_distance, &light_pdf);
					glm::vec3 Llocal = to_local(glm::quat{shaderdata_stream.T.w[ID], shaderdata_stream.T.x[ID], shaderdata_stream.T.y[ID], shaderdata_stream.T.z[ID]}, L);
					if (Llocal.z < 0.0f) [[unlikely]] continue; //sample below hemisphere
					glm::vec3 radiance = scene.material[light_prim.material_ID].emission * glm::vec3{in->throughput.r[ID], in->throughput.g[ID], in->throughput.b[ID]};
					radiance *= reinterpret_cast<Closure<static_cast<ClosureType>(BRDF)>*>(&shaderdata_stream.closure[ID])->eval(
						Llocal,
						glm::vec3{shaderdata_stream.V.x[ID], shaderdata_stream.V.y[ID], shaderdata_stream.V.z[ID]}
					);
					light_pdf *= light_selection_pdf; //uniformly sampling a light with pdf: 1/light_count
					float brdf_pdf = reinterpret_cast<Closure<static_cast<ClosureType>(BRDF)>*>(&shaderdata_stream.closure[ID])->pdf(Llocal);
					radiance *= powerHeuristic_over_f(light_pdf, brdf_pdf);
					if (std::max(std::max(radiance.r, radiance.g), radiance.b) <= 0.0f) [[unlikely]] continue;
					ray_stream.shadow_rays.dir.x     [shadow_index] = L.x;
					ray_stream.shadow_rays.dir.y     [shadow_index] = L.y;
					ray_stream.shadow_rays.dir.z     [shadow_index] = L.z;
					ray_stream.shadow_rays.p.x       [shadow_index] = shaderdata_stream.P.x[ID];
					ray_stream.shadow_rays.p.y       [shadow_index] = shaderdata_stream.P.y[ID];
					ray_stream.shadow_rays.p.z       [shadow_index] = shaderdata_stream.P.z[ID];
					ray_stream.shadow_rays.tfar      [shadow_index] = light_distance;
					ray_stream.shadow_rays.radiance.r[shadow_index] = radiance.r;
					ray_stream.shadow_rays.radiance.g[shadow_index] = radiance.g;
					ray_stream.shadow_rays.radiance.b[shadow_index] = radiance.b;
					ray_stream.flags.has_shadowray.set(ID);
					++shadow_index;
				}
				/*---------------------------------
				------- SHADOW RAY TRACING -------- //Profiling: 7%
				---------------------------------*/
				scene.acceleration_structure.Traverse_shadow<StreamSize>(ray_stream.shadow_rays, shadow_index);
				//--- Accumulation //Profiling: 2%
				for (size_t i = miss_count, shadow_ID = 0; i < active_rays; i++) {
					const int32_t& ID = ray_stream.RayID[i];
					if (ray_stream.flags.has_shadowray.test(ID)) {
						if (!ray_stream.shadow_rays.occluded.test(shadow_ID)) {
							in->radiance.r[ID] += ray_stream.shadow_rays.radiance.r[shadow_ID];
							in->radiance.g[ID] += ray_stream.shadow_rays.radiance.g[shadow_ID];
							in->radiance.b[ID] += ray_stream.shadow_rays.radiance.b[shadow_ID];
						}
						++shadow_ID; //pdep shadow stream data
					}
				}
			#endif
				/*---------------------------------
				----- EMISSIVE PRIMITIVE HIT ----- //Profiling: 6%
				---------------------------------*/
				if (MIS && bounce > 0) {
					for (size_t ID = 0; ID < active_rays; ID++) {
						if (shaderdata_stream.is_emissive.test(ID)) {
							glm::vec3 throughput{in->throughput.r[ID], in->throughput.g[ID], in->throughput.b[ID]};
							const auto& light_prim = scene.acceleration_structure.prims[ray_stream.hit.primID[ID]];
							const float& radius2 = light_prim.radius_sq;
						#if false
							glm::vec3 Wc = light_prim.position - glm::vec3{in->p.x[ID], in->p.y[ID], in->p.z[ID]};
							float center_dist2 = glm::dot(Wc, Wc);
						#else //alternative version via law of cosines to not require having ray origin at this point, comprobado en houdini q es equivalente
							const float& depth = ray_stream.hit.tfar[ID];
							const float& NdotV = shaderdata_stream.V.z[ID];
							float center_dist2 = depth * (depth + NdotV * (2.0f * sqrt(radius2))) + radius2;
						#endif
							float weight = powerHeuristic( //note (brdf / brdf_pdf) term for this light is already accounted for in throughput at this point all that remains is the balance heuristic
								in->pdf[ID],                                           //brdf_pdf: pdf of previous bounce sampling what has now hit a light
								light_selection_pdf * spherePdf(radius2, center_dist2) //light_pdf: pdf of sampling this light directly
							);
							throughput *= weight;
							const glm::vec3& em = scene.material[ray_stream.hit.matID[ID]].emission;
							in->radiance.r[ID] += throughput.r * em.r;
							in->radiance.g[ID] += throughput.g * em.g;
							in->radiance.b[ID] += throughput.b * em.b;
						}
					}
				} else {
					for (size_t ID = 0; ID < active_rays; ID++) {
						if (shaderdata_stream.is_emissive.test(ID)) {
							const glm::vec3& em = scene.material[ray_stream.hit.matID[ID]].emission;
							in->radiance.r[ID] += em.r;
							in->radiance.g[ID] += em.g;
							in->radiance.b[ID] += em.b;
						}
					}
				}
				/*---------------------------------
				----- BRDF SAMPLING - BOUNCE ----- //Profiling: 13%
				---------------------------------*/
				size_t output_index = 0; //after loop indicates count of new rays
				if (bounce < Policy.max_bounces - 1) [[likely]] {
					for (size_t i = 0; i < hit_count; i++) {
						const int32_t& ID = ray_stream.RayID[miss_count + i];
						const uint32_t& seed = ray_stream.seed[in->pixelID[ID]];
						uint32_t rng_state = hash_2d(accumulations, ray_stream.seed[in->pixelID[ID]] + bounce * 2 + 1);
						Sample bsdf_sample;
						{//Sample BSDF

							const std::array<float, 2> brdf_samples{rand_unit_float(&rng_state), rand_unit_float(&rng_state)};

							//FALTA IMPLEMENTAR PDF DE GGX
							reinterpret_cast<Closure<static_cast<ClosureType>(BRDF)>*>(&shaderdata_stream.closure[ID])->sample(
								bsdf_sample,
								glm::vec3{shaderdata_stream.V.x[ID], shaderdata_stream.V.y[ID], shaderdata_stream.V.z[ID]},
								brdf_samples.data()
							);
						}
						glm::vec3 throughput{in->throughput.r[ID], in->throughput.g[ID], in->throughput.b[ID]};
						throughput *= bsdf_sample.estimator; //estimator = NdotL * brdf_eval / pdf
						{//Russian roulette
							float q = 1.0f - std::max(throughput.r, std::max(throughput.g, throughput.b));
							if (rand_unit_float(&rng_state) < q) {
								ray_stream.flags.termination.set(ID);
								continue;
							}
							throughput *= 1.0f / std::max(FLT_EPSILON, 1.0f - q);
						}
						{//Transform sample to worldspace
							bsdf_sample.dir = to_world(glm::quat{shaderdata_stream.T.w[ID], shaderdata_stream.T.x[ID], shaderdata_stream.T.y[ID], shaderdata_stream.T.z[ID]}, bsdf_sample.dir);
						}
						out->p.x         [output_index] = shaderdata_stream.P.x[ID];
						out->p.y         [output_index] = shaderdata_stream.P.y[ID];
						out->p.z         [output_index] = shaderdata_stream.P.z[ID];
						out->dir.x       [output_index] = bsdf_sample.dir.x;
						out->dir.y       [output_index] = bsdf_sample.dir.y;
						out->dir.z       [output_index] = bsdf_sample.dir.z;
						out->throughput.r[output_index] = throughput.r;
						out->throughput.g[output_index] = throughput.g;
						out->throughput.b[output_index] = throughput.b;
						out->radiance.r  [output_index] = in->radiance.r[ID];
						out->radiance.g  [output_index] = in->radiance.g[ID];
						out->radiance.b  [output_index] = in->radiance.b[ID];
						out->pixelID     [output_index] = in->pixelID[ID];
						out->pdf         [output_index] = reinterpret_cast<Closure<static_cast<ClosureType>(BRDF)>*>(&shaderdata_stream.closure[ID])->pdf(bsdf_sample.dir);
						output_index++;
					}
				}
					/*---------------------------------
					---------  MISS SHADER ----------- //Profiling: has_ambient ? 10% : ~0%
					---------------------------------*/
				for (size_t i = 0; i < miss_count; i++) {
					ray_stream.flags.termination.set(ray_stream.RayID[i]);
				}
				if (has_ambient) {
					for (size_t i = 0; i < miss_count; i++) {
						const int32_t& ID = ray_stream.RayID[i];
						const int32_t& px = in->pixelID[ID];
						glm::vec3 sky_value = scene.sky(in->dir.x[ID], in->dir.y[ID], in->dir.z[ID]);
						in->radiance.r[ID] += in->throughput.r[ID] * sky_value.r;
						in->radiance.g[ID] += in->throughput.r[ID] * sky_value.g;
						in->radiance.b[ID] += in->throughput.r[ID] * sky_value.b;
					}
				}
				/*---------------------------------
				---------  ACCUMULATION ----------- //Profiling: 3%
				---------------------------------*/
				for (size_t ID = 0; ID < active_rays; ID++) {
					if (!ray_stream.flags.termination.test(ID)) continue;
					const int32_t& px = in->pixelID[ID];
					output_color.r[px] += in->radiance.r[ID];
					output_color.g[px] += in->radiance.g[ID];
					output_color.b[px] += in->radiance.b[ID];
				}
				active_rays = output_index;
			}
		}, concurrency::auto_partitioner{});
	}

	void Render() requires (AccumulationBuckets == 5) {
		if (accumulations % 5) return; //only every 5th accumulation has equal number of samples per bucket which is assumed below when weighting all buckets equally
		concurrency::parallel_for(0u, static_cast<uint32_t>(framebuffer.size() / TileSize), [this, 
			scale = Vec8f{scene.camera.exp / static_cast<float>(accumulations / AccumulationBuckets)}
		] (const uint32_t LaunchIndex) {
			/*---------------------------------
			-- MEDIAN OF MEANS & TONEMAPPING -- //Profiling: 2%
			---------------------------------*/
			static constexpr size_t stride = TileSize / 8; //512 bytes
			const size_t dst_row_stride = 4 * (width - TileRoot); //next row but back to the beginning of the tile; 4 floats per pixel
			const Vec8f* __restrict src = (Vec8f*)&accumulator[LaunchIndex].color;
			float* __restrict dst = (float*)&framebuffer[TileRoot * ((LaunchIndex / h_tiles) * width + (LaunchIndex % h_tiles))];
			for (size_t i = 0; i < TileRoot; i++, dst+=dst_row_stride) {
				for (size_t j = 0; j < TileRoot; j+= 8, dst+=32, src++) {
				#define MEDIAN true
				#if MEDIAN == true
					/*                                          Bucket 0         Bucket 1         Bucket 2         Bucket 3          Bucket 4 */
					Vec8f r = scale * median<Vec8f>(/*R:*/src[stride * 0], src[stride * 3], src[stride * 6], src[stride *  9], src[stride * 12]);
					Vec8f g = scale * median<Vec8f>(/*G:*/src[stride * 1], src[stride * 4], src[stride * 7], src[stride * 10], src[stride * 13]);
					Vec8f b = scale * median<Vec8f>(/*B:*/src[stride * 2], src[stride * 5], src[stride * 8], src[stride * 11], src[stride * 14]);
				#else //AVERAGE OF BUCKETS
					Vec8f r = scale * 0.2f * (/*R:*/src[stride * 0] + src[stride * 3] + src[stride * 6] + src[stride *  9] + src[stride * 12]);
					Vec8f g = scale * 0.2f * (/*G:*/src[stride * 1] + src[stride * 4] + src[stride * 7] + src[stride * 10] + src[stride * 13]);
					Vec8f b = scale * 0.2f * (/*B:*/src[stride * 2] + src[stride * 5] + src[stride * 8] + src[stride * 11] + src[stride * 14]);
				#endif
					tonemapping(r, g, b);
					r = _mm256_permutevar8x32_ps(r, _mm256_setr_epi32(0,4,2,6,1,5,3,7));  //{r0 r4 r2 r6 | r1 r5 r3 r7}
					g = _mm256_permutevar8x32_ps(g, _mm256_setr_epi32(4,0,6,2,5,1,7,3));  //{g4 g0 g6 g2 | g5 g1 g7 g3}
					b = _mm256_permutevar8x32_ps(b, _mm256_setr_epi32(2,6,0,4,3,7,1,5));  //{b2 b6 b0 b4 | b3 b7 b1 b5}
					const __m256 a  = _mm256_set1_ps(1.0f);//         6,2,4,0,7,3,5,1     //{a6 a2 a4 a0 | a7 a3 a5 a1}		
					const __m256 rg = _mm256_blend_ps(r, g, 0b1010'1010);                 //{r0 g0 r2 g2 | r1 g1 r3 g3}
					const __m256 ba = _mm256_blend_ps(b, a, 0b1010'1010);                 //{b2 a2 b0 a0 | b3 a3 b1 a1}
					const __m256 gr = _mm256_blend_ps(g, r, 0b1010'1010);                 //{g4 r4 g6 r6 | g5 r5 g7 r7}
					const __m256 ab = _mm256_blend_ps(a, b, 0b1010'1010);                 //{a6 b6 a4 b4 | a7 b7 a5 b5}
					_mm256_stream_ps(dst +  0, _mm256_blend_ps  (rg, ba, 0b1100'1100  )); //{r0 g0 b0 a0 | r1 g1 b1 a1}
					_mm256_stream_ps(dst +  8, _mm256_shuffle_ps(rg, ba, 0b01'00'11'10)); //{r2 g2 b2 a2 | r3 g3 b3 a3}
					_mm256_stream_ps(dst + 16, _mm256_shuffle_ps(gr, ab, 0b10'11'00'01)); //{r4 g4 b4 a4 | r5 g5 b5 a5}
					_mm256_stream_ps(dst + 24, _mm256_shuffle_ps(gr, ab, 0b00'01'10'11)); //{r6 g6 b6 a6 | r7 g7 b7 a7}
				}
			}
		}, concurrency::auto_partitioner{});
		output_frame->SetData(framebuffer.data());
	}
};