#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <concepts>
#include <span>
#include <bitset>
#include <memory_resource>

#include "Core.hpp"
#include "Primitives.hpp"
#include "DataStructures.hpp"
#include "DataStreams.hpp"

template<typename Primitive>
struct BoundingVolumeHierarchy {
	struct alignas(32) Node {
		using Self = Node;
		using Scalar = float;
		using Vector = glm::vec<3, Scalar>;

		alignas(16) Vector min_bound;
		uint32_t first_id;
		alignas(16) Vector max_bound;
		uint32_t prim_count;

		Node() :
			min_bound(+std::numeric_limits<Scalar>::max()), max_bound(-std::numeric_limits<Scalar>::max()), first_id(0), prim_count(0) {}
		Node(const Vector& lower, const Vector& upper, uint32_t first_id = 0, uint32_t primitive_count = 0) :
			min_bound(lower), max_bound(upper), first_id(first_id), prim_count(primitive_count) {}
		explicit Node(const Vector& point) : Node(point, point) {}
		explicit Node(const Primitive& prim) : Node(prim.bounds().min, prim.bounds().max) {}

		Self& operator|=(const Self& other) {
			min_bound = min(min_bound, other.min_bound);
			max_bound = max(max_bound, other.max_bound);
			return *this;
		}
		friend Self operator|(Self a, const Self& b) { return a |= b; }

		bool is_leaf() const {
			return prim_count != 0;
		}
		Vector diagonal() const {
			return max_bound - min_bound;
		}
		size_t largest_axis() const {
			const Vector diag = diagonal();
			const float* d = (float*)&diag;
			size_t ret = 0;
			for (size_t i = 1; i < 3; ++i) if (d[ret] < d[i]) ret = i;
			return ret;
		}
		Vector centroid() const {
			return (max_bound + min_bound) * static_cast<Scalar>(0.5);
		}
		Scalar half_area() const {
			const Vector d = diagonal();
			Scalar area = static_cast<Scalar>(0.0);
			int32_t i = 2;
			for (Scalar accum = d[i--]; i > 0; i--) {
				area += d[i] * accum;
				accum += d[i];
			}
			return area;
		}
	};

	struct SplitHeuristic {
		size_t log_cluster_size = 0; //log of the size of primitive clusters in base 2 
		float cost_ratio = 1.0f; //cost of intersecting a node (a ray-box intersection) over the cost of intersecting a primitive

		float prim_count(size_t size) const {
			return static_cast<float>((size + (1ull << log_cluster_size) - 1) >> log_cluster_size);
		}
		float leaf_cost(size_t size, float bbox_half_area) const {
			return bbox_half_area * prim_count(size);
		}
		float non_split_cost(size_t size, float bbox_half_area) const {
			return bbox_half_area * (prim_count(size) - cost_ratio);
		}
	};

	std::vector<Node> nodes;
	std::vector<Primitive> prims; //could be some sort of precomputed prim for faster traversal

	template <class Self> auto&& root(this Self&& self) { return std::forward<Self>(self).front(); }
	BoundingVolumeHierarchy(){}
	BoundingVolumeHierarchy(std::span<const Primitive> primitives, SplitHeuristic heuristic = SplitHeuristic{}) {
		struct StackFrame { size_t ID, begin, count; };
		struct Split { size_t pos, axis; float cost; };
		//-----------------ALLOCATION----------------------
		const size_t primnum = primitives.size();
		const size_t node_estimate = 2 * (primnum + 1);//upper estimate

		const size_t buffer_size = primnum * (3 * sizeof(uint32_t) + sizeof(Node) + sizeof(Node::Vector) + sizeof(float));	//6GB buffer for 100M prims
		auto framebuffer = std::make_unique_for_overwrite<char[]>(buffer_size);
		std::pmr::monotonic_buffer_resource temporary_mbr{framebuffer.get(), buffer_size};

		std::pmr::vector<uint32_t>primIDs{&temporary_mbr}; primIDs.resize(primnum * 3);
		std::pmr::vector<Node> bboxes{&temporary_mbr}; bboxes.resize(primnum);
		std::pmr::vector<Node::Vector> centroids{&temporary_mbr}; centroids.resize(primnum);
		std::pmr::vector<float>accum_cost{&temporary_mbr}; accum_cost.resize(primnum);
		std::vector<bool>marks; marks.resize(primnum);
		prims.resize(primnum);
		nodes.reserve(node_estimate);
		//----------------------------------------------------
		auto reduce_bboxes = [&](size_t from, size_t to) -> Node {
			Node res{};
			for (size_t i = from; i < to; ++i) res |= bboxes[primIDs[i]];
			return res;
		};

		{ // Make and sort bboxes
			std::ranges::transform(primitives, bboxes.begin(), [](auto&& val) { return Node{val}; });
			std::ranges::transform(bboxes, centroids.begin(), [](auto&& val) { return val.centroid(); });
			for (size_t axis = 0; axis < 3; ++axis) {
				std::span axis_ids{primIDs.begin() + (axis * primnum), primnum};
				std::ranges::iota(axis_ids, 0);
				std::ranges::sort(axis_ids, [&](uint32_t a, uint32_t b) { return ((float*)&(centroids[a]))[axis] < ((float*)&(centroids[b]))[axis]; });
			}
		}

		nodes.emplace_back(std::accumulate(bboxes.begin(), bboxes.end(), Node{}, std::bit_or<Node>{})); //root node

		Stack<StackFrame, 64> stack;
		stack.push(StackFrame{0, 0, primnum});

		while (!stack.empty()) {
			auto item = stack.pop();
			Node& node = nodes[item.ID];
			if (item.count <= 1 /*LEAF SIZE*/) {
				node.first_id = static_cast<uint32_t>(item.begin);
				node.prim_count = static_cast<uint32_t>(item.count);
				continue;
			}
			const size_t first_child = nodes.size();
			node.first_id = static_cast<uint32_t>(first_child);
			//------------------------
			const size_t begin = item.begin;
			const size_t end = item.begin + item.count;
			//------------------------
			Split best_split{begin + (item.count + 1) / 2, node.largest_axis(), heuristic.non_split_cost(item.count, node.half_area())};
			{// Find best split
				for (size_t axis = 0; axis < 3; ++axis) {
					size_t first_right = 0;
					// Sweep from the right to the left, computing the partial SAH cost
					auto right_bbox = Node{};
					for (size_t i = end - 1; i > begin;) {
						auto right_cost = 0.0f;
						for (; i > i - std::min(i - begin, 32ull); --i) {
							right_bbox |= bboxes[primIDs[axis * primnum + i]];
							accum_cost[i] = right_cost = heuristic.leaf_cost(end - i, right_bbox.half_area());
						}
						// Every `chunk_size` elements, check that we are not above the maximum cost
						if (right_cost > best_split.cost) {
							first_right = i;
							break;
						}
					}
					// Sweep from the left to the right, computing the full cost
					auto left_bbox = Node{};
					for (size_t i = begin; i < end - 1; i++) {
						left_bbox |= bboxes[primIDs[axis * primnum + i]];
						if (i < first_right) break;
						auto left_cost = heuristic.leaf_cost(i + 1 - begin, left_bbox.half_area());
						if (left_cost > best_split.cost) break;
						if (auto cost = left_cost + accum_cost[i + 1]; cost < best_split.cost) best_split = Split{i + 1, axis, cost};
					}
				}
			}
			{// Partition bboxes
				for (size_t i = begin; i < best_split.pos; ++i) marks[primIDs[best_split.axis * primnum + i]] = true;
				for (size_t i = best_split.pos; i < end; ++i) marks[primIDs[best_split.axis * primnum + i]] = false;
				for (size_t axis = 0; axis < 3; ++axis) {
					if (axis == best_split.axis) continue;
					std::stable_partition(
						primIDs.begin() + axis * primnum + begin,
						primIDs.begin() + axis * primnum + end,
						[&](size_t ID) { return marks[ID]; }
					);
				}
			}
			{// Recursion
				struct Range {size_t begin, end;};
				const std::array<Range, 2> ranges{Range{begin, best_split.pos}, Range{best_split.pos, end}};
				const std::array<Node, 2> children{reduce_bboxes(ranges[0].begin, ranges[0].end), reduce_bboxes(ranges[1].begin, ranges[1].end)};

				size_t sort_area = static_cast<size_t>(children[0].half_area() < children[1].half_area());
				size_t sort_size = static_cast<size_t>(ranges[0].end - ranges[0].begin < ranges[1].end - ranges[1].begin);
				size_t combined = sort_area ^ sort_size;

				nodes.push_back(children[sort_area]);
				nodes.push_back(children[1 - sort_area]);
				stack.push(StackFrame{first_child +      combined , ranges[sort_size].begin, ranges[sort_size].end - ranges[sort_size].begin});
				stack.push(StackFrame{first_child + (1 - combined), ranges[1 - sort_size].begin, ranges[1 - sort_size].end - ranges[1 - sort_size].begin});
			}
		}
		nodes.shrink_to_fit();
		{ //reorder primtives by the primids to remove the indirection
			for (size_t i = 0; i < primnum; i++) {
				prims[i] = primitives[primIDs[i]];
			}
		}
	}

	template<size_t N>
	struct AABB_acceleration_struct {
		struct {
			float x[N];
			float y[N];
			float z[N];
		} m, n;
		float t[N];
	};
	
	//taking the entire accel reference and an index is not great but with SOA layout I dont see any other way
	template<size_t N>
	[[msvc::forceinline]] static bool test_AABB(const AABB_acceleration_struct<N>& accel, const Node& node, size_t index) {
		float lo = node.min_bound.x * accel.m.x[index] - accel.n.x[index];
		float hi = node.max_bound.x * accel.m.x[index] - accel.n.x[index];
		float tmin = glm::max(1e-4f, glm::min(lo, hi));
		float tmax = glm::min(accel.t[index], glm::max(lo, hi));
		lo = node.min_bound.y * accel.m.y[index] - accel.n.y[index];
		hi = node.max_bound.y * accel.m.y[index] - accel.n.y[index];
		tmin = glm::max(tmin, glm::min(lo, hi));
		tmax = glm::min(tmax, glm::max(lo, hi));
		lo = node.min_bound.z * accel.m.z[index] - accel.n.z[index];
		hi = node.max_bound.z * accel.m.z[index] - accel.n.z[index];
		tmin = glm::max(tmin, glm::min(lo, hi));
		tmax = glm::min(tmax, glm::max(lo, hi));
		return tmax >= tmin;
	}

	template<size_t N>
	[[msvc::forceinline]] void intersect_prims(typename const RayStream<N>::Buffer& in, typename RayStream<N>::Hit& out, size_t begin_ray, size_t end_ray, size_t begin_prim, size_t end_prim) const requires(std::is_same_v<Primitive, Sphere>) {
	
	#define SIMD_INTERSECT true	
		for (size_t prim_ID = begin_prim; prim_ID < end_prim; prim_ID++) {
			size_t ID = begin_ray;
		#if SIMD_INTERSECT
			const __m256i broadcast_primID = _mm256_set1_epi32(static_cast<int32_t>(prim_ID)); //alternatively increment in the loop like prim_ID
			const __m256 broadcast_pos[3]{
				_mm256_set1_ps((float)prims[prim_ID].position.x),
				_mm256_set1_ps((float)prims[prim_ID].position.y),
				_mm256_set1_ps((float)prims[prim_ID].position.z)
			};
			const __m256 broadcast_radiussq = _mm256_set1_ps((float)prims[prim_ID].radius_sq);
			for (; (ID + 7) < end_ray; ID += 8) {
				__m256 temp_x = _mm256_sub_ps(broadcast_pos[0], _mm256_load_ps((const float*)&in.p.x[ID]));
				__m256 b = _mm256_mul_ps(_mm256_load_ps((const float*)&in.dir.x[ID]), temp_x);
				__m256 discriminant = _mm256_fnmadd_ps(temp_x, temp_x, broadcast_radiussq);
				__m256 temp_y = _mm256_sub_ps(broadcast_pos[1], _mm256_load_ps((const float*)&in.p.y[ID]));
				b = _mm256_fmadd_ps(_mm256_load_ps((const float*)&in.dir.y[ID]), temp_y, b);
				discriminant = _mm256_fnmadd_ps(temp_y, temp_y, discriminant);
				__m256 temp_z = _mm256_sub_ps(broadcast_pos[2], _mm256_load_ps((const float*)&in.p.z[ID]));
				b = _mm256_fmadd_ps(_mm256_load_ps((const float*)&in.dir.z[ID]), temp_z, b);
				discriminant = _mm256_fnmadd_ps(temp_z, temp_z, discriminant);
				discriminant = _mm256_fmadd_ps(b, b, discriminant);
				if (_mm256_movemask_ps(discriminant) == 0xFFFFFFFF) continue; //all discriminants are negative
				discriminant = _mm256_sqrt_ps(discriminant);
				__m256 dist = _mm256_sub_ps(b, discriminant);
				dist = _mm256_blendv_ps(dist, _mm256_add_ps(b, discriminant), dist); //if dist < 0.0 => blend b + d instead
				__m256i mask = std::bit_cast<__m256i>(_mm256_andnot_ps(_mm256_or_ps(discriminant, dist), _mm256_cmp_ps(dist, _mm256_load_ps((const float*)&out.tfar[ID]), 1))); //dist < tfar & ~(dist < 0 || discr < 0)
				_mm256_maskstore_ps((float*)&out.tfar[ID], mask, dist);
				_mm256_maskstore_epi32((int32_t*)&out.primID[ID], mask, broadcast_primID);
			}
		#endif
			for (; ID < end_ray; ID++) {
				float b = 0.0f;
				float discriminant = prims[prim_ID].radius_sq;
				for (size_t dim = 0; dim < 3; dim++)
				{
					float temp = ((float*)&prims[prim_ID].position)[dim] - ((float*)&in.p)[dim * N + ID];
					b += ((float*)&in.dir)[dim * N + ID] * temp;
					discriminant -= temp * temp;
				}
				discriminant += b * b;
				if (discriminant < 0.0f) continue;
				discriminant = sqrt(discriminant);
				float dist = (b >= discriminant ? b - discriminant : b + discriminant);
				if (dist < 0.0f || dist >= out.tfar[ID]) continue;
				out.tfar[ID] = dist;
				out.primID[ID] = static_cast<int32_t>(prim_ID);
			}
		}
	}

	template<size_t N>
	[[msvc::forceinline]] void intersect_prims_shadow(typename RayStream<N>::ShadowStream& in, size_t begin_ray, size_t end_ray, size_t begin_prim, size_t end_prim) const requires(std::is_same_v<Primitive, Sphere>) {
		for (size_t ID = begin_ray; ID < end_ray; ID++) {
			for (size_t prim_ID = begin_prim; prim_ID < end_prim; prim_ID++) {
				glm::vec3 P{prims[prim_ID].position.x - in.p.x[ID], prims[prim_ID].position.y - in.p.y[ID], prims[prim_ID].position.z - in.p.z[ID]};
				float b = glm::dot(glm::vec3{in.dir.x[ID], in.dir.y[ID], in.dir.z[ID]}, P);
				float discriminant = b * b - glm::dot(P, P) + prims[prim_ID].radius_sq;
				if (discriminant < 0.0f) continue;
				discriminant = sqrt(discriminant);
				float dist = (b >= discriminant ? b - discriminant : b + discriminant);
				if (dist < 0.0f || dist >= in.tfar[ID]) continue;
				in.occluded.set(ID);
				break; //GO TO NEXT RAY BECAUSE THIS RAY IS ALREADY SHADOWED BY AT LEAST ONE PRIM => OJO AL ORDEN DE LOS LOOPS
			}
		}
	}

#define USEBVH false

	template<size_t N>
	void Traverse(typename const RayStream<N>::Buffer& in, typename RayStream<N>::Hit& out, size_t size) const {
	#if !USEBVH
		intersect_prims<N>(in, out, 0, size, 0, prims.size());
		for (size_t i = 0; i < size; i++) {
			if (auto primID = out.primID[i]; primID >= 0) {
				out.matID[i] = static_cast<int32_t>(prims[primID].material_ID);
			}
		}
		return;
	#else
		struct StackFrame { size_t ID; size_t head; };
		Stack<StackFrame, 64> stack;
		StackFrame frame{0, 0};

		//Sorting rays based on direction/origin? => "Ray Binning"

		AABB_acceleration_struct<N> AABB_accel;
		for (size_t i = 0; i < size; i++){
			const int32_t ID = static_cast<int32_t>(i); // ray_stream.RayID[i] if sorted in some manner
			AABB_accel.n.x[i] = in.p.x[ID] * (AABB_accel.m.x[i] = 1.0f / in.dir.x[ID]);
			AABB_accel.n.y[i] = in.p.y[ID] * (AABB_accel.m.y[i] = 1.0f / in.dir.y[ID]);
			AABB_accel.n.z[i] = in.p.z[ID] * (AABB_accel.m.z[i] = 1.0f / in.dir.z[ID]);
			AABB_accel.t[i] = out.tfar[ID];
		}

		for (;;) { restart:
			const Node& node = nodes[frame.ID];
			for (; frame.head < size; frame.head++) {
				if (test_AABB<N>(AABB_accel, node, frame.head)) {
					if (!nodes[frame.ID].is_leaf()) {
						bool firstChild = 0;//traversalOrder(frame.node, raypack); <<< ============ TODO ===================
						stack.push(StackFrame{(size_t)node.first_id + (1ull - firstChild), frame.head});
						frame.ID = (size_t)node.first_id + firstChild;
						goto restart;
					}
					intersect_prims<N>(in, out, frame.head, size, node.first_id, node.first_id + node.prim_count);
					break;
				}
			}
			if (stack.empty()) {
				for (size_t i = 0; i < size; i++) {
					if (auto primID = out.primID[i]; primID >= 0) {
						out.matID[i] = static_cast<int32_t>(prims[primID].material_ID);
					}
				}
				return;
			}
			frame = stack.pop();
		}
	#endif
	}

	template<size_t N>
	void Traverse_shadow(typename RayStream<N>::ShadowStream& in, size_t size) const {
	#if !USEBVH
		intersect_prims_shadow<N>(in, 0, size, 0, prims.size());
		return;
	#else
		struct StackFrame {
			size_t ID; size_t head;
		};
		Stack<StackFrame, 64> stack;
		StackFrame frame{0, 0};

		AABB_acceleration_struct<N> AABB_accel;
		for (size_t i = 0; i < size; i++) {
			const int32_t ID = static_cast<int32_t>(i);
			AABB_accel.n.x[i] = in.p.x[ID] * (AABB_accel.m.x[i] = 1.0f / in.dir.x[ID]);
			AABB_accel.n.y[i] = in.p.y[ID] * (AABB_accel.m.y[i] = 1.0f / in.dir.y[ID]);
			AABB_accel.n.z[i] = in.p.z[ID] * (AABB_accel.m.z[i] = 1.0f / in.dir.z[ID]);
			AABB_accel.t[i] = in.tfar[ID];
		}

		for (;;) {
		restart:
			const Node& node = nodes[frame.ID];
			for (; frame.head < size; frame.head++) {
				if (test_AABB<N>(AABB_accel, node, frame.head)) {
					if (!nodes[frame.ID].is_leaf()) {
						bool firstChild = 0;//traversalOrder(frame.node, raypack);
						stack.push(StackFrame{(size_t)node.first_id + (1ull - firstChild), frame.head});
						frame.ID = (size_t)node.first_id + firstChild;
						goto restart;
					}
					intersect_prims_shadow<N>(in, frame.head, size, node.first_id, node.first_id + node.prim_count);
					break;
				}
			}
			if (stack.empty()) {
				return;
			}
			frame = stack.pop();
		}
	#endif
	}
};