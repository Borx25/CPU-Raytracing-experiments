#include <array>
#include <random>
#include <iostream>
#include <fstream>
#include <format>

#include "Image.h"
#include <GLFW/glfw3.h>
#include "stb_image.h"
#include "backends/imgui_impl_glfw.h"

#include "App.hpp"
#include "Core.hpp"
#include "DataStructures.hpp"
#include "Renderer.hpp"
#include "Camera.hpp"
#include "Scene.hpp"


using namespace glm;

class RaytracingApp : public App {
	enum class Scenes {
		Default,
		BVH_test,
		BRDF_test,
		White_Furnace
	};

	static constexpr Scenes SceneSelection = Scenes::Default;
public:
	RaytracingApp(std::string_view name, int32_t width, int32_t height) : App(name, width, height) {
		if constexpr(SceneSelection == Scenes::Default){
			//CAMERA
			scene.camera = Camera{{-0.2, 0.3, 1}, {0.1, -0.4, -1}, 1, 1, 40.0, 0.0f, 16.0f, 1.0f};
			//floor
			scene.material.push_back(Material{});
			scene.material.back().albedo = vec3{1.0f};
			scene.material.back().F0 = vec3{0.8f};
			scene.material.back().F80 = vec3{0.9f};
			scene.material.back().roughness = 0.2f;
			scene.geometry.push_back(Sphere{vec3{0.3, -1.47,0.0f}, 1.5f * 1.5f, (int32_t)scene.material.size() - 1});
			//light
			scene.material.push_back(Material{});
			scene.material.back().emission = 0.1f * vec3{25.0f, 25.0f, 200.0f};
			scene.material.back().albedo = vec3{1.0f};
			scene.material.back().roughness = 1.0f;
			scene.geometry.push_back(Sphere{vec3{0.29999, 0.0801,0.0f}, 0.05f * 0.05f, (int32_t)scene.material.size() - 1});

			scene.material.push_back(Material{});
			scene.material.back().emission = 0.1f * vec3{150.0f, 150.0f, 150.0f};
			scene.material.back().albedo = vec3{1.0f};
			scene.material.back().roughness = 1.0f;
			scene.geometry.push_back(Sphere{vec3{0.3302, 0.36165,0.7119}, 0.05f * 0.05f, (int32_t)scene.material.size() - 1});

			scene.material.push_back(Material{});
			scene.material.back().emission = vec3{200.0f, 17.0f, 25.0f};
			scene.material.back().albedo = vec3{1.0f};
			scene.material.back().roughness = 1.0f;
			scene.geometry.push_back(Sphere{vec3{-0.4857, -0.0242 ,-0.41383}, 0.05f * 0.05f, (int32_t)scene.material.size() - 1});
			//GEOMETRY
			scene.material.push_back(Material{});
			scene.material.back().albedo = vec3{0.793, 0.793, 0.664};
			scene.material.back().F0 = vec3{0.04f};
			scene.material.back().F80 = vec3{0.5f};
			scene.material.back().roughness = 0.85f;
			scene.geometry.push_back(Sphere{vec3{0.3, 1.7 ,0.0f}, 1.5f * 1.5f, (int32_t)scene.material.size() - 1});

			scene.material.push_back(Material{});
			scene.material.back().albedo = vec3{0.05f, 0.05f, 0.05f};
			scene.material.back().F0 = vec3{0.03f, 0.03f, 0.03f};
			scene.material.back().F80 = vec3{0.5f, 0.5f, 0.5f};
			scene.material.back().transmission = vec3{0.95, 0.95, 0.95};
			scene.material.back().IOR_minus_one = 0.44f;
			scene.material.back().roughness = 0.05;
			scene.geometry.push_back(Sphere{vec3{0.018, 0.022f ,0.07}, 0.02f * 0.02f, (int32_t)scene.material.size() - 1});

			scene.material.push_back(Material{});
			scene.material.back().albedo = vec3{1.0};
			scene.material.back().F0 = vec3{0.944, 0.776, 0.373};
			scene.material.back().F80 = vec3{0.8f, 0.8f, 0.6f};
			scene.material.back().roughness = 0.15;
			scene.geometry.push_back(Sphere{vec3{-0.037, 0.022f ,0.00}, 0.03f * 0.03f, (int32_t)scene.material.size() - 1});

			scene.material.push_back(Material{});
			scene.material.back().albedo = vec3{1.0f};
			scene.material.back().F0 = vec3{0.076288, 0.077375, 0.078887};
			scene.material.back().F80 = vec3{0.47990, 0.48028, 0.48080};
			scene.material.back().transmission = vec3{0.670, 0.764, 0.855};
			scene.material.back().IOR_minus_one = 0.762f;
			scene.material.back().roughness = 0.1;
			scene.geometry.push_back(Sphere{vec3{-0.0846, -0.0334 ,0.283}, 0.012f * 0.012f, (int32_t)scene.material.size() - 1});

			scene.material.push_back(Material{});
			scene.material.back().albedo = vec3{1.0f};
			scene.material.back().F0 = vec3{0.04f};
			scene.material.back().F80 = vec3{0.5f};
			scene.material.back().roughness = 0.8f;
			scene.geometry.push_back(Sphere{vec3{0.03863, -0.00788 ,0.2835}, 0.012f * 0.012f, (int32_t)scene.material.size() - 1});
			//SKY
			scene.sky.ambient_color = vec3{0.0f, 0.0f, 0.0f}; //ambient
		} else if constexpr (SceneSelection == Scenes::BVH_test) {
			//CAMERA
			scene.camera = Camera{{0, 60, 300}, {0, 0, -1}};
			//MATERIALS
			//GEOMETRY
			std::mt19937 engine(static_cast<uint32_t>(0x8aa214e404d15a07));
			std::uniform_int_distribution<int> mat_dist(0, static_cast<int>(scene.material.size()) - 1);
			std::uniform_real_distribution<float> y_dist(0.0f, 100.0f);
			std::uniform_real_distribution<float> xz_dist(-100.0f, 100.0f);
			std::uniform_real_distribution<float> radius_dist(0.3f, 20.0f);
			static constexpr int num_spheres = 255;
			scene.geometry.reserve(num_spheres);
			for (size_t i = 0; i < num_spheres; i++){
				float r = radius_dist(engine);
				scene.geometry.push_back(Sphere{
					vec3{xz_dist(engine), y_dist(engine), xz_dist(engine)},
					r * r,
					mat_dist(engine)
				});
			}
			scene.sky.ambient_color = vec3{1.0f, 1.0f, 1.0f}; //ambient
		} else if constexpr (SceneSelection == Scenes::BRDF_test) {
			static constexpr int32_t gradations = 10;
			const float cam_offset = static_cast<float>(gradations) * 2.8f;
			scene.camera = Camera{{0, 0, cam_offset}, {0, 0, -1}};

			//floor
			scene.material.push_back(Material{});
			scene.material.back().albedo = vec3{0.1f};
			scene.material.back().roughness = 1.0f;
			scene.geometry.push_back(Sphere{vec3{0.0f, -1001.0f,0.0f}, 1000.0f * 1000.0f, (int32_t)scene.material.size() - 1});

			//light
			scene.material.push_back(Material{});
			scene.material.back().emission = vec3{100.0f, 100.0f, 100.0f};
			scene.geometry.push_back(Sphere{vec3{0.0f, 10.0f,0.0f}, 5.0f, (int32_t)scene.material.size() - 1});

			for (int32_t i = 0; i < gradations; i++) {
				const float t = static_cast<float>(i) / static_cast<float>(gradations - 1);
				const float x = static_cast<float>(i * 2 - gradations) * 1.25f + 1.0f;

				scene.material.push_back(Material{});
				scene.geometry.push_back(Sphere{vec3{x, static_cast<float>(i) * 0.1f ,0.0f}, 1.0f, (int32_t)scene.material.size() - 1});

				Material& mat = scene.material.back();

				enum class Properties {
					Roughness,
					RoughnessDiffuse,
					IOR_reflection,
					IOR_refraction,
					Roughness_glass,
					Absorption,
					Absorption_roughness,
					Refraction_to_diffuse,
				};

				switch (Properties::Roughness)
				{
				case Properties::Roughness: 
					mat.F0 = vec3{1.0f, 1.0f, 1.0f};
					mat.F80 = vec3{1.0f, 1.0f, 1.0f};
					mat.albedo = vec3{0.0f, 0.0f, 0.0f};
					mat.roughness = t;
					break;
				case Properties::RoughnessDiffuse:
					mat.F0 = vec3{0.04f, 0.04f, 0.04f};
					mat.F80 = vec3{0.5f, 0.5f, 0.5f};
					mat.albedo = vec3{0.75f, 0.25f, 0.25f};
					mat.roughness = t;
					break;
				case Properties::IOR_reflection:
					mat.F0 = vec3{0.04f, 0.04f, 0.04f};
					mat.F80 = vec3{0.5f, 0.5f, 0.5f};
					mat.albedo = vec3{0.7, 0.5, 0.3};
					mat.IOR_minus_one = t; //Ior 1 to 2
					break;
				case Properties::IOR_refraction:
					mat.F0 = vec3{0.04f, 0.04f, 0.04f};
					mat.F80 = vec3{0.5f, 0.5f, 0.5f};
					mat.transmission = vec3{0.95, 0.95, 0.95};
					mat.IOR_minus_one = t * 0.5f; //Ior 1 to 2
					break;
				case Properties::Roughness_glass:
					mat.F0 = vec3{0.04f, 0.04f, 0.04f};
					mat.F80 = vec3{0.5f, 0.5f, 0.5f};
					mat.transmission = vec3{0.95, 0.95, 0.95};
					mat.IOR_minus_one = 0.1f;
					mat.roughness = t;
					break;
				case Properties::Absorption:
					mat.F0 = vec3{0.04f, 0.04f, 0.04f};
					mat.F80 = vec3{0.5f, 0.5f, 0.5f};
					mat.transmission = glm::mix(vec3{0.95, 0.95, 0.95}, vec3{0, 0.95, 0.95}, t);
					mat.IOR_minus_one = 0.1f;
					mat.roughness = 0.0f;
					break;
				case Properties::Absorption_roughness:
					mat.F0 = vec3{0.04f, 0.04f, 0.04f};
					mat.F80 = vec3{0.5f, 0.5f, 0.5f};
					mat.transmission = vec3{0.0f, 0.95, 0.95};
					mat.IOR_minus_one = 0.1f;
					mat.roughness = t;
					break;
				case Properties::Refraction_to_diffuse:
					mat.F0 = vec3{0.04f, 0.04f, 0.04f};
					mat.F80 = vec3{0.5f, 0.5f, 0.5f};
					mat.albedo = glm::mix(vec3{0.0f, 0.0f, 0.0f}, vec3{0, 0.95, 0.95}, t);
					mat.transmission = glm::mix(vec3{0.95, 0.95, 0.95}, vec3{0, 0, 0}, t);
					mat.IOR_minus_one = 0.0f;
					mat.roughness = 0.0f;
					break;
				default: break;
				}
			}
			scene.sky.ambient_color = vec3{1.0f, 1.0f, 1.0f}; //ambient
		} else if constexpr (SceneSelection == Scenes::White_Furnace) {
			scene.camera = Camera{{0, 0, 3}, {0, 0, -1}};
			scene.material.push_back(Material{vec3{1.0f, 1.0f, 1.0f}});
			scene.geometry.push_back(Sphere{vec3{0, 0, 0}, 1.0f, 0});
			scene.sky.ambient_color = vec3{1.0f, 1.0f, 1.0f}; //ambient
		}

		scene.sky.hdri_data = stbi_loadf("C:\\Users\\borja\\source\\repos\\Raytracing\\env.hdr", &scene.sky.hdri_width, &scene.sky.hdri_height, &scene.sky.hdri_channels, 4);
		if (scene.sky.hdri_data == nullptr) {
			std::cout << stbi_failure_reason();
			std::terminate();
		}
		scene.sky.hdri_fwidth = static_cast<float>(scene.sky.hdri_width - 1);
		scene.sky.hdri_fheight = static_cast<float>(scene.sky.hdri_height - 1);

		scene.acceleration_structure = decltype(scene.acceleration_structure){scene.geometry};
		scene.lighting_acceleration = decltype(scene.lighting_acceleration){scene.geometry, scene.material};

		glfwSetWindowUserPointer(WindowHandle, this);
		glfwSetKeyCallback(WindowHandle, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
			RaytracingApp* ptr = (RaytracingApp*)glfwGetWindowUserPointer(window);
			if (action == GLFW_PRESS) {
				switch (key)
				{
				case GLFW_KEY_TAB:
					ptr->viewport_only = !ptr->viewport_only;
					break;
				case GLFW_KEY_P:
					ptr->scene.camera.view = default_view;
					ptr->renderer.ResetAccumulator();
					break;
				case GLFW_KEY_SPACE:
					glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
					ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
					ptr->moving_camera = true;
					break;
				case GLFW_KEY_F5:
					std::string filename = std::format("C:\\Users\\borja\\source\\repos\\Raytracing\\screenshot_{}.hdr", ptr->screenshot_index++);
					Image::Store(filename.c_str(), (float*)ptr->renderer.framebuffer.data(), ptr->renderer.width, ptr->renderer.height);
					break;
				}
			} else if (action == GLFW_RELEASE) {
				switch (key)
				{
				case GLFW_KEY_SPACE:
					glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
					ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
					ptr->moving_camera = false;
					break;
				}
			}
			ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
		});
		glfwSetMouseButtonCallback(WindowHandle, [](GLFWwindow* window, int button, int action, int mods) {
			if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
				RaytracingApp* ptr = (RaytracingApp*)glfwGetWindowUserPointer(window);
				static constexpr size_t TileRoot = decltype(ptr->renderer)::TileRoot;

				double x_f, y_f;
				glfwGetCursorPos(window, &x_f, &y_f);
				int32_t x = static_cast<int32_t>(x_f);
				int32_t y = static_cast<int32_t>(y_f);
				if (0 <= x && x < static_cast<int32_t>(ptr->viewport_width) && 0 <= y && y < static_cast<int32_t>(ptr->viewport_height)) {
					y = ptr->viewport_height - y;
					RayStream<8> depth_ray{};
					depth_ray.hit.matID[0] = -1;
					depth_ray.hit.primID[0] = -1;
					depth_ray.hit.tfar[0] = FLT_MAX;
					auto* const __restrict raygen_buffer = const_cast<RayStream<8>::Buffer*>(depth_ray.path.input);

					static constexpr float no_pixel_jitter[2]{0.5f, 0.5f};
					const auto [orig, dir] = ptr->scene.camera.generate_ray(x, y, no_pixel_jitter);
					raygen_buffer->dir.x[0] = dir.x;
					raygen_buffer->dir.y[0] = dir.y;
					raygen_buffer->dir.z[0] = dir.z;
					raygen_buffer->p.x[0] = orig.x;
					raygen_buffer->p.y[0] = orig.y;
					raygen_buffer->p.z[0] = orig.z;

					ptr->scene.acceleration_structure.Traverse<8>(*depth_ray.path.input, depth_ray.hit, 1);
					ptr->scene.camera.projection.focus_distance = depth_ray.hit.matID[0] > -1 ? depth_ray.hit.tfar[0] : INFINITY;
					ptr->renderer.ResetAccumulator();
				}

			}
			ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
		});
	}

	static inline const View default_view{{0, 60, 300}, {0, 0, -1}};

	void Update(float ts) {
		float frame_time_ms = this->FrameTime * 1e3f;
		frame_time_moving_average = glm::mix(frame_time_moving_average, frame_time_ms, (2.0f / static_cast<float>(64 + 1)));
		frame_time_buffer.push_back(frame_time_ms);
		Vec2d prev_mousepos = mousepos;
		glfwGetCursorPos(WindowHandle, (double*)&mousepos, (double*)&mousepos + 1);
		if (!moving_camera) return;
   		vec3 move_vector{
			  static_cast<float>(glfwGetKey(WindowHandle, GLFW_KEY_D) - glfwGetKey(WindowHandle, GLFW_KEY_A)),  //Right
			  static_cast<float>(glfwGetKey(WindowHandle, GLFW_KEY_E) - glfwGetKey(WindowHandle, GLFW_KEY_Q)),  //Up
			  static_cast<float>(glfwGetKey(WindowHandle, GLFW_KEY_S) - glfwGetKey(WindowHandle, GLFW_KEY_W))}; //Backward
		bool reset = false;
		if (move_vector != vec3{0.0, 0.0, 0.0}) {
			scene.camera.TranslateLocal(camera_speed.movement * ts * move_vector);
			reset = true;
		}
		float roll = static_cast<float>(glfwGetKey(WindowHandle, GLFW_KEY_F) - glfwGetKey(WindowHandle, GLFW_KEY_R)); //Roll
		if (_mm_movemask_pd(mousepos != prev_mousepos) != 0 || roll != 0.0f) {
			float amount = ts * camera_speed.rotation;
			Vec4f delta = 0.01f * amount * Vec4f{_mm_cvtpd_ps(mousepos - prev_mousepos)};
			scene.camera.RotateLocal({delta[1], delta[0], 0.25f * amount * roll});
			reset = true;
		}
		if(reset) renderer.ResetAccumulator();
	}

	enum class SceneUpdate {
		Geometry,
		Material,
		Light,
		Ambient,
		Camera
	};

	enum class UpdateTracker : int32_t {
		Null = 0,
		Accumulator = -1, //track any update
		BVH = 1u << static_cast<int32_t>(SceneUpdate::Geometry),//track geometry changes only
		Lighting = (1u << static_cast<int32_t>(SceneUpdate::Light)) | (1u << static_cast<int32_t>(SceneUpdate::Geometry)) | (1u << static_cast<int32_t>(SceneUpdate::Material))
	};

	INLINE friend constexpr UpdateTracker& operator|=(UpdateTracker& tracker, UpdateTracker update){
		return tracker = static_cast<UpdateTracker>(static_cast<int32_t>(tracker) | static_cast<int32_t>(update));
	}
	INLINE friend constexpr UpdateTracker operator&(UpdateTracker a, UpdateTracker b) {
		return static_cast<UpdateTracker>(static_cast<int32_t>(a) & static_cast<int32_t>(b));
	}
	INLINE friend constexpr bool operator!(UpdateTracker tracker) {
		return !static_cast<bool>(tracker);
	}


	void UIRender() {
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));
		ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoCollapse);
		{
			uint32_t width = static_cast<uint32_t>(ImGui::GetContentRegionAvail().x);
			uint32_t height = static_cast<uint32_t>(ImGui::GetContentRegionAvail().y);
			static constexpr size_t tile_requirement = decltype(renderer)::RequiredTiling();
			if (width % tile_requirement != 0 || height % tile_requirement != 0) { //Pad to tile
				width = (width + (tile_requirement - 1)) & ~(tile_requirement - 1);
				height = (height + (tile_requirement - 1)) & ~(tile_requirement - 1);
				ImGui::SetWindowSize(ImVec2{static_cast<float>(width), static_cast<float>(height)});
			}
			if (width != viewport_width || height != viewport_height) {
				viewport_width = width; viewport_height = height;
				renderer.Resize(viewport_width, viewport_height);
				scene.camera.Resize(viewport_width, viewport_height);
			}
			if (auto& frame = renderer.GetFrame()) { //RENDER
				renderer.Accumulate();
				renderer.Render();
				ImGui::Image(frame->GetDescriptorSet(), ImVec2{(float)frame->GetWidth(), (float)frame->GetHeight()}, ImVec2(0, 1), ImVec2(1, 0));
			}
		}
		ImGui::End();
		ImGui::PopStyleVar();

		if (viewport_only) return;
		UpdateTracker scene_changes = UpdateTracker::Null;
		ImGui::Begin("Info");
		{
			std::array<float, decltype(frame_time_buffer)::Size> plot_data;
			float max_current = 0.0f;
			for (size_t i = 0; i < plot_data.size(); i++) {
				plot_data[i] = frame_time_buffer[i];
				max_current = max(max_current, plot_data[i]);
			}; 
			if (plot_ceiling / max_current > 5.0f) plot_ceiling = 0.0f;
			plot_ceiling = glm::mix(glm::mix(plot_ceiling, max_current, (2.0f / static_cast<float>(256 + 1))), 2.0f * frame_time_moving_average, 0.5);
			ImGui::PushItemWidth(-1.0f);
			float frames_per_ms = 1.0f / frame_time_moving_average;
			int fps = static_cast<int>(1e3f * frames_per_ms);
			float Msamples_per_s = static_cast<float>(viewport_width * viewport_height * renderer.Policy.samples_per_pixel) * (1e-3f * frames_per_ms);
			std::string info = std::format("[{} X {}] : {:.2f}ms : {}fps : {:.2f}Msamples/s", viewport_width, viewport_height, frame_time_moving_average, fps, Msamples_per_s);
			ImGui::PlotLines("##frame_plot", plot_data.data(), static_cast<int>(plot_data.size()), 0, nullptr, 0.0, plot_ceiling, ImVec2{0.0,100});
			ImGui::SetCursorPosX((ImGui::GetWindowSize().x - ImGui::GetFontSize() * info.size() / 2) / 2);
			ImGui::Text(info.c_str());
			ImGui::Separator();
			ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.5f);
			ImGui::DragFloat("##CamRotation", &camera_speed.rotation, 0.01f, 0.0f, INFINITY, "Camera Rotation: %.3f");
			ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
			ImGui::DragFloat("##CamMovement", &camera_speed.movement, 0.01f, 0.0f, INFINITY, "Camera Movement: %.3f");
			ImGui::PopItemWidth();
			scene_changes |= static_cast<UpdateTracker>(ImGui::DragFloat("##CamFocalLength", &scene.camera.projection.focal_length, 0.01f, 0.0f, INFINITY, "Camera Focal Length (mm): %.3f") << static_cast<int32_t>(SceneUpdate::Camera));
			scene_changes |= static_cast<UpdateTracker>(ImGui::DragFloat("##CamFNumber", &scene.camera.projection.f_number, 0.01f, 0.0f, INFINITY, "Camera F-number: %.3f") << static_cast<int32_t>(SceneUpdate::Camera));
			scene_changes |= static_cast<UpdateTracker>(ImGui::DragFloat("##CamFocusDistance", &scene.camera.projection.focus_distance, 0.5f, 0.0f, INFINITY, "Camera Focus Distance: %.3f") << static_cast<int32_t>(SceneUpdate::Camera));
			scene_changes |= static_cast<UpdateTracker>(ImGui::DragFloat("##CamExposure", &scene.camera.exp, 0.001f, 0.0f, INFINITY, "Camera Exposure: %.3f") << static_cast<int32_t>(SceneUpdate::Camera));
			scene.camera.projection.UpdateLens();
			ImGui::PopItemWidth();
			ImGui::Text(std::format("Camera pos: [{}, {}, {}]", scene.camera.view.pos.x, scene.camera.view.pos.y, scene.camera.view.pos.z).c_str());
			ImGui::Text(std::format("Spheres: {}, BVH size: {}", scene.geometry.size(), scene.acceleration_structure.nodes.size()).c_str());
		}
		ImGui::End();
		ImGui::Begin("Scene Editor");
		{
			auto centered_text = [](const char* label) {
				ImGui::SetCursorPosX((ImGui::GetWindowSize().x - ImGui::GetFontSize() * strlen(label) / 2) / 2);
				ImGui::Text(label);
			};
			auto array_selector = [&](const char* ID, const char* format, int32_t* index, size_t arr_size) -> bool {
				ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.40f, 0.40f, 0.40f, 0.50f));
				bool ret = ImGui::SliderInt(ID, index, 0, static_cast<int32_t>(arr_size) - 1, format);
				ImGui::PopStyleColor();
				return ret;
			};
			auto color_picker = [&](const char* label, glm::vec3& color, SceneUpdate category, ImGuiColorEditFlags flags = 0) {
				vec3 temp = pow(color, vec3{1.0f / 2.2f});
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{temp.r, temp.g, temp.b, 0.5f});
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{temp.r, temp.g, temp.b, 1.0f});
				ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f - temp.r, 1.0f - temp.b, 1.0f - temp.g, 1.0f});
				if (ImGui::Button(label, ImVec2(ImGui::CalcItemWidth(), 25.0)))
					active_colorpicker = (active_colorpicker == label ? nullptr : label);
				ImGui::PopStyleColor(3);

				if (active_colorpicker == label) {
					bool open = true;
					ImGui::PushItemWidth(-1.0f);
					bool edited = ImGui::ColorPicker3(label, (float*)&temp, flags | ImGuiColorEditFlags_Float | ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoLabel);
					ImGui::PopItemWidth();
					scene_changes |= static_cast<UpdateTracker>(edited << static_cast<int32_t>(category));
					if (edited) {
						color = pow(temp, vec3{2.2f});
						//Sanitize value
						color.x = isnan(color.x) || isinf(color.x) ? 0.0f : color.x;
						color.y = isnan(color.x) || isinf(color.y) ? 0.0f : color.y;
						color.z = isnan(color.x) || isinf(color.z) ? 0.0f : color.z;
					}
					if (!open) active_colorpicker = nullptr;
				}
			};

			ImGui::PushItemWidth(-1.0f);

			centered_text("Geometry");
			if (scene.geometry.size() > 0) {
				array_selector("##geo_index", "Active sphere: %d", &active_geo_index, scene.geometry.size());
				scene_changes |= static_cast<UpdateTracker>(ImGui::DragFloat3("##geo_pos", (float*)&scene.geometry[active_geo_index].position, 0.001f) << static_cast<int32_t>(SceneUpdate::Geometry));
				float radius = sqrt(scene.geometry[active_geo_index].radius_sq);
				scene_changes |= static_cast<UpdateTracker>(ImGui::DragFloat("##geo_radius", &radius, 0.001f, 0.0f, INFINITY, "Radius: %.3f") << static_cast<int32_t>(SceneUpdate::Geometry));
				scene.geometry[active_geo_index].radius_sq = radius * radius;
				scene_changes |= static_cast<UpdateTracker>(array_selector("##geo_matID", "Material ID: %d", &scene.geometry[active_geo_index].material_ID, scene.material.size()) << static_cast<int32_t>(SceneUpdate::Material));
			}
			ImGui::Separator();

			centered_text("Materials");
			if (scene.material.size() > 0) {
				array_selector("##mat_index", "Active material: %d", &active_material_index, scene.material.size());
				color_picker("Albedo", scene.material[active_material_index].albedo, SceneUpdate::Material);
				color_picker("Reflection", scene.material[active_material_index].F0, SceneUpdate::Material);
				color_picker("Edge tint", scene.material[active_material_index].F80, SceneUpdate::Material);
				color_picker("Emission", scene.material[active_material_index].emission, SceneUpdate::Material, ImGuiColorEditFlags_HDR);
				color_picker("Transmission", scene.material[active_material_index].transmission, SceneUpdate::Material);
				//scene_changes |= static_cast<UpdateTracker>(ImGui::DragFloat("##emissive_power", &scene.material[active_material_index].emissive_intensity, 0.01f, 0.0f, INFINITY, "Emissive power: %.3f") << static_cast<int32_t>(SceneUpdate::Material));
				ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.40f, 0.40f, 0.40f, 0.50f));
				scene_changes |= static_cast<UpdateTracker>(ImGui::SliderFloat("##roughness", &scene.material[active_material_index].roughness, 0.0f, 1.0f, "Roughness: %.3f") << static_cast<int32_t>(SceneUpdate::Material));
				scene_changes |= static_cast<UpdateTracker>(ImGui::SliderFloat("##ior", &scene.material[active_material_index].IOR_minus_one, 0.0f, 10.0f, "IOR (minus one): %.3f") << static_cast<int32_t>(SceneUpdate::Material));
				ImGui::PopStyleColor();
			}
			ImGui::Separator();

			//centered_text("Lights");
			//if (scene.lighting.size() > 0) {
			//	array_selector("##light_index", "Active light: %d", &active_light_index, scene.lighting.size());
			//	scene_changes |= static_cast<UpdateTracker>(ImGui::DragFloat3("##light_pos", (float*)&scene.lighting[active_light_index].position, 0.05f) << static_cast<int32_t>(SceneUpdate::Light));
			//	color_picker("Light", scene.lighting[active_light_index].color, SceneUpdate::Light);
			//	scene_changes |= static_cast<UpdateTracker>(ImGui::DragFloat("##light_power", &scene.lighting[active_light_index].power, 1.0f, 0.0f, INFINITY, "Power: %.3f") << static_cast<int32_t>(SceneUpdate::Light));
			//	float radius = sqrt(scene.lighting[active_light_index].radius_sq);
			//	scene_changes |= static_cast<UpdateTracker>(ImGui::DragFloat("##light_radius", &radius, 0.01f, 0.0f, INFINITY, "Radius: %.3f") << static_cast<int32_t>(SceneUpdate::Light));
			//	scene.lighting[active_light_index].radius_sq = radius * radius;
			//}
			//ImGui::Separator();

			centered_text("Global");
			color_picker("Ambient", scene.sky.ambient_color, SceneUpdate::Ambient, ImGuiColorEditFlags_HDR);

			ImGui::PopItemWidth();
		}
		ImGui::End();
		if (!!(scene_changes & UpdateTracker::BVH)) scene.acceleration_structure = decltype(scene.acceleration_structure){scene.geometry};
		if (!!(scene_changes & UpdateTracker::Lighting)) scene.lighting_acceleration = decltype(scene.lighting_acceleration){scene.geometry, scene.material};
		if (!!(scene_changes & UpdateTracker::Accumulator)) renderer.ResetAccumulator();
	}
	Scene scene;
	//Renderer<SceneSelection == Scenes::BVH_test> renderer{scene};
	Renderer<> renderer{scene};
	private:
	//viewport
	bool viewport_only = false;
	uint32_t viewport_width = 0, viewport_height = 0;
	//scene editor
	int32_t active_geo_index = 0;
	int32_t active_material_index = 0;
	//int32_t active_light_index = 0;
	const char* active_colorpicker = nullptr;
	//performance plot
	float frame_time_moving_average = 0.0f;
	CyclicBuffer<float, 64> frame_time_buffer{};
	float plot_ceiling = 0.0f;
	//Flying camera
	bool moving_camera = false;
	Vec2d mousepos{0.0, 0.0};
	struct {
		float movement = 1.0f;
		float rotation = 1.0f;
	} camera_speed;
	uint32_t screenshot_index = 0;
};

int main(int argc, char** argv)
{
	RaytracingApp app("CPURaytracer", 1920, 1080);
	app.Run();
}