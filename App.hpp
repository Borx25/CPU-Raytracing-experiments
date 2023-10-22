#include <string_view>
#include <functional>

#include "vulkan/vulkan.h"
#include "imgui.h"

struct GLFWwindow;

template<typename T>
concept ValidApp = requires (T&& app) {
	app.UIRender();
	app.Update(.0f);
};

void check_vk_result(VkResult err);

struct App {
	App(std::string_view name, int32_t width, int32_t height);
	~App();
	bool IsRunning() const;
	void Close();

	double GetTime() const;
	static VkPhysicalDevice GetPhysicalDevice();
	static VkDevice GetDevice();
	static VkCommandBuffer GetCommandBuffer(bool begin);
	static void FlushCommandBuffer(VkCommandBuffer commandBuffer);
	static void SubmitResourceFree(std::function<void()>&& func);

	GLFWwindow* WindowHandle = nullptr;
	double FrameTime = 0.0;
private:
	void PollEvents() const;
	void FrameInit() const;
	void FrameRender(ImDrawData*) const;
	void FramePresent() const;

	std::string_view WindowName;
	double LastTime = 0.0, CurrentTime = 0.0;
public:
	template <ValidApp Self> void Run(this Self&& self) {
		ImGuiIO& io = ImGui::GetIO();
		while (self.IsRunning()) {
			self.PollEvents();
			self.Update(self.FrameTime);
			{ //RENDER FRAME
				self.FrameInit();
				ImGui::NewFrame();
				{ //IMGUI
					static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None | ImGuiDockNodeFlags_AutoHideTabBar;

					// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
					// because it would be confusing to have two docking targets within each others.
					ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
					//window_flags |= ImGuiWindowFlags_MenuBar; //Enable if rendering a menu bar

					//this is what fixes the dockspace to the main viewport
					const ImGuiViewport* viewport = ImGui::GetMainViewport();
					ImGui::SetNextWindowPos(viewport->WorkPos);
					ImGui::SetNextWindowSize(viewport->WorkSize);
					ImGui::SetNextWindowViewport(viewport->ID);

					ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
					ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
					window_flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
					window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
					window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

					// When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
					// and handle the pass-thru hole, so we ask Begin() to not render a background.
					if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
						window_flags |= ImGuiWindowFlags_NoBackground;

					// Important: note that we proceed even if Begin() returns false (aka window is collapsed).
					// This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
					// all active windows docked into it will lose their parent and become undocked.
					// We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
					// any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
					ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
					ImGui::Begin(self.WindowName.data(), nullptr, window_flags);


					ImGui::PopStyleVar();

					ImGui::PopStyleVar(2);

					//Submit the DockSpace
					ImGuiIO& io = ImGui::GetIO();
					if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
					{
						ImGuiID dockspace_id = ImGui::GetID("VulkanAppDockspace");
						ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
					}

					self.UIRender();

					ImGui::End();
				}

				// Rendering
				ImGui::Render();
				ImDrawData* draw_data = ImGui::GetDrawData();
				const bool main_is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
				if (!main_is_minimized) self.FrameRender(draw_data);

				// Update and Render additional Platform Windows
				if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
				{
					ImGui::UpdatePlatformWindows();
					ImGui::RenderPlatformWindowsDefault();
				}

				// Present Main Platform Window
				if (!main_is_minimized) self.FramePresent();
			}
			self.CurrentTime = self.GetTime();
			self.FrameTime = self.CurrentTime - self.LastTime;
			self.LastTime = self.CurrentTime;
		}
	}
};

