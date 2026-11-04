#include "RmlUi_Backend.h"
#include "RmlUi_Platform_SDL.h"
#include "RmlUi_Renderer_WGPU.h"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/Log.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "sdl3webgpu.h"
#include "wgpu.h"

#include <cstdio>
#include <memory>
#include <utility>

/**
    Global data used by this backend.

    Lifetime governed by the calls to Backend::Initialize() and Backend::Shutdown().
 */
struct BackendData {
	BackendData(SDL_Window* window, WGPUInstance instance, WGPUSurface surface, WGPUAdapter adapter, WGPUDevice device, WGPUQueue queue,
		WGPUTextureFormat format) :
		system_interface(window), render_interface(device, queue, format), window(window), instance(instance), surface(surface), adapter(adapter),
		device(device), queue(queue) {};

	SystemInterface_SDL system_interface;
	RenderInterface_WGPU render_interface;
	TextInputMethodEditor_SDL text_input_method_editor;

	SDL_Window* window = nullptr;
	WGPUInstance instance = nullptr;
	WGPUSurface surface = nullptr;
	WGPUAdapter adapter = nullptr;
	WGPUDevice device = nullptr;
	WGPUQueue queue = nullptr;
	WGPUSurfaceConfiguration config{};

	bool running = true;
	bool frame_ready = false;

	RawKeyDownCallback raw_key_down_callback;
	RawTextEditingCallback raw_text_editing_callback;

	// Scancodes whose initial key-down was consumed by the raw key callback. Repeat events for those
	// keys must also be suppressed so they do not leak into the RmlUi context.
	bool suppressed_key_down[SDL_SCANCODE_COUNT] = {};

	// Last values pushed into the RmlUi context, used to only react to actual window changes.
	int synced_width = 0;
	int synced_height = 0;
	float synced_pixel_density = 0.f;

	WGPUSurfaceTexture surface_texture{};
	WGPUTextureView surface_view = nullptr;
	WGPUCommandEncoder command_encoder = nullptr;
};
static Rml::UniquePtr<BackendData> data;

namespace {

void OnWGPULog(WGPULogLevel level, WGPUStringView message, void* /*userdata*/)
{
	const char* level_str = "info";
	switch (level)
	{
	case WGPULogLevel_Error: level_str = "error"; break;
	case WGPULogLevel_Warn: level_str = "warn"; break;
	case WGPULogLevel_Info: level_str = "info"; break;
	case WGPULogLevel_Debug: level_str = "debug"; break;
	case WGPULogLevel_Trace: level_str = "trace"; break;
	default: level_str = "unknown"; break;
	}
	std::fprintf(stderr, "[wgpu] [%s] %.*s\n", level_str, int(message.length), message.data);
}

struct AdapterRequestResult {
	WGPUAdapter adapter = nullptr;
};

void OnRequestAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void* /*userdata2*/)
{
	if (status == WGPURequestAdapterStatus_Success)
	{
		static_cast<AdapterRequestResult*>(userdata1)->adapter = adapter;
	}
	else
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to request a WebGPU adapter: %.*s", int(message.length), message.data);
	}
}

struct DeviceRequestResult {
	WGPUDevice device = nullptr;
};

void OnRequestDevice(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void* /*userdata2*/)
{
	if (status == WGPURequestDeviceStatus_Success)
	{
		static_cast<DeviceRequestResult*>(userdata1)->device = device;
	}
	else
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to request a WebGPU device: %.*s", int(message.length), message.data);
	}
}

WGPUTextureFormat ChooseSurfaceFormat(const WGPUSurfaceCapabilities& capabilities)
{
	// Prefer a non-sRGB format so that colors are written as-is, matching the premultiplied sRGB values used by RmlUi.
	for (size_t i = 0; i < capabilities.formatCount; i++)
	{
		if (capabilities.formats[i] == WGPUTextureFormat_BGRA8Unorm || capabilities.formats[i] == WGPUTextureFormat_RGBA8Unorm)
			return capabilities.formats[i];
	}
	return capabilities.formatCount > 0 ? capabilities.formats[0] : WGPUTextureFormat_Undefined;
}

void ConfigureSurface(BackendData& backend)
{
	int width = 0, height = 0;
	SDL_GetWindowSizeInPixels(backend.window, &width, &height);
	if (width <= 0 || height <= 0)
		return;

	backend.config.width = width;
	backend.config.height = height;
	wgpuSurfaceConfigure(backend.surface, &backend.config);
}

} // namespace

bool Backend::Initialize(const char* window_name, int width, int height, bool allow_resize)
{
	RMLUI_ASSERT(!data);

	SDL_SetMainReady();
	SDL_SetHint(SDL_HINT_IME_IMPLEMENTED_UI, "composition");
	wgpuSetLogCallback(OnWGPULog, nullptr);
	wgpuSetLogLevel(WGPULogLevel_Warn);

	if (!SDL_Init(SDL_INIT_VIDEO))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "SDL error on init: %s", SDL_GetError());
		return false;
	}

	// Submit click events when focusing the window.
	SDL_SetHint(SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH, "1");
	// Touch events are handled natively, no need to generate synthetic mouse events for touch devices.
	SDL_SetHint(SDL_HINT_TOUCH_MOUSE_EVENTS, "0");

	SDL_WindowFlags window_flags = SDL_WINDOW_HIGH_PIXEL_DENSITY;
	if (allow_resize)
		window_flags |= SDL_WINDOW_RESIZABLE;

	SDL_Window* window = SDL_CreateWindow(window_name, width, height, window_flags);
	if (!window)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "SDL error on create window: %s", SDL_GetError());
		SDL_Quit();
		return false;
	}

	{
		int window_width = 0, window_height = 0;
		int pixel_width = 0, pixel_height = 0;
		SDL_GetWindowSize(window, &window_width, &window_height);
		SDL_GetWindowSizeInPixels(window, &pixel_width, &pixel_height);
		Rml::Log::Message(Rml::Log::LT_INFO, "SDL window size: %dx%d (drawable %dx%d, pixel density %.2f, display scale %.2f)", window_width,
			window_height, pixel_width, pixel_height, SDL_GetWindowPixelDensity(window), SDL_GetWindowDisplayScale(window));
	}

	WGPUInstance instance = wgpuCreateInstance(nullptr);
	if (!instance)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create the WebGPU instance.");
		SDL_DestroyWindow(window);
		SDL_Quit();
		return false;
	}

	WGPUSurface surface = SDL_GetWGPUSurface(instance, window);
	if (!surface)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create the WebGPU surface from the SDL window.");
		wgpuInstanceRelease(instance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return false;
	}

	AdapterRequestResult adapter_result;
	WGPURequestAdapterOptions adapter_options{};
	adapter_options.compatibleSurface = surface;
	WGPURequestAdapterCallbackInfo adapter_callback{};
	adapter_callback.callback = OnRequestAdapter;
	adapter_callback.userdata1 = &adapter_result;
	wgpuInstanceRequestAdapter(instance, &adapter_options, adapter_callback);

	if (!adapter_result.adapter)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to acquire a WebGPU adapter.");
		wgpuSurfaceRelease(surface);
		wgpuInstanceRelease(instance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return false;
	}

	DeviceRequestResult device_result;
	WGPURequestDeviceCallbackInfo device_callback{};
	device_callback.callback = OnRequestDevice;
	device_callback.userdata1 = &device_result;

	WGPUFeatureName required_features[1];
	WGPUDeviceDescriptor device_descriptor{};
	WGPUFeatureName polygon_mode_line = static_cast<WGPUFeatureName>(WGPUNativeFeature_PolygonModeLine);
	if (wgpuAdapterHasFeature(adapter_result.adapter, polygon_mode_line))
	{
		required_features[0] = polygon_mode_line;
		device_descriptor.requiredFeatureCount = 1;
		device_descriptor.requiredFeatures = required_features;
	}
	wgpuAdapterRequestDevice(adapter_result.adapter, &device_descriptor, device_callback);

	if (!device_result.device)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to acquire a WebGPU device.");
		wgpuAdapterRelease(adapter_result.adapter);
		wgpuSurfaceRelease(surface);
		wgpuInstanceRelease(instance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return false;
	}

	WGPUQueue queue = wgpuDeviceGetQueue(device_result.device);
	if (!queue)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to acquire the WebGPU queue.");
		wgpuDeviceRelease(device_result.device);
		wgpuAdapterRelease(adapter_result.adapter);
		wgpuSurfaceRelease(surface);
		wgpuInstanceRelease(instance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return false;
	}

	WGPUSurfaceCapabilities capabilities{};
	wgpuSurfaceGetCapabilities(surface, adapter_result.adapter, &capabilities);

	const WGPUTextureFormat surface_format = ChooseSurfaceFormat(capabilities);
	if (surface_format == WGPUTextureFormat_Undefined || capabilities.alphaModeCount == 0)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "The WebGPU surface does not provide any usable configuration.");
		wgpuSurfaceCapabilitiesFreeMembers(capabilities);
		wgpuQueueRelease(queue);
		wgpuDeviceRelease(device_result.device);
		wgpuAdapterRelease(adapter_result.adapter);
		wgpuSurfaceRelease(surface);
		wgpuInstanceRelease(instance);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return false;
	}

	WGPUSurfaceConfiguration config{};
	config.device = device_result.device;
	config.format = surface_format;
	config.usage = WGPUTextureUsage_RenderAttachment;
	config.presentMode = WGPUPresentMode_Fifo;
	config.alphaMode = capabilities.alphaModes[0];

	wgpuSurfaceCapabilitiesFreeMembers(capabilities);

	data = Rml::MakeUnique<BackendData>(window, instance, surface, adapter_result.adapter, device_result.device, queue, surface_format);
	data->config = config;
	ConfigureSurface(*data);

	Rml::SetTextInputHandler(&data->text_input_method_editor);

	return true;
}

void Backend::Shutdown()
{
	RMLUI_ASSERT(data);

	data->render_interface.Shutdown();

	if (data->surface_view)
	{
		wgpuTextureViewRelease(data->surface_view);
		data->surface_view = nullptr;
	}
	if (data->surface_texture.texture)
	{
		wgpuTextureRelease(data->surface_texture.texture);
		data->surface_texture.texture = nullptr;
	}
	if (data->command_encoder)
	{
		wgpuCommandEncoderRelease(data->command_encoder);
		data->command_encoder = nullptr;
	}

	wgpuSurfaceRelease(data->surface);
	wgpuQueueRelease(data->queue);
	wgpuDeviceRelease(data->device);
	wgpuAdapterRelease(data->adapter);
	wgpuInstanceRelease(data->instance);
	wgpuSetLogCallback(nullptr, nullptr);

	SDL_Window* window = data->window;
	data.reset();

	SDL_DestroyWindow(window);
	SDL_Quit();
}

Rml::SystemInterface* Backend::GetSystemInterface()
{
	RMLUI_ASSERT(data);
	return &data->system_interface;
}

Rml::RenderInterface* Backend::GetRenderInterface()
{
	RMLUI_ASSERT(data);
	return &data->render_interface;
}

bool Backend::ProcessEvents(Rml::Context* context, KeyDownCallback key_down_callback, bool power_save)
{
	RMLUI_ASSERT(data && context);

	auto GetKey = [](const SDL_Event& event) { return event.key.key; };
	auto GetDisplayScale = []() { return SDL_GetWindowDisplayScale(data->window); };
	constexpr auto event_quit = SDL_EVENT_QUIT;
	constexpr auto event_key_down = SDL_EVENT_KEY_DOWN;
	constexpr auto event_text_editing = SDL_EVENT_TEXT_EDITING;
	bool has_event = false;
	bool suppress_text_input = false;

	bool result = data->running;
	data->running = true;

	SDL_Event ev;
	if (power_save)
		has_event = SDL_WaitEventTimeout(&ev, static_cast<int>(Rml::Math::Min(context->GetNextUpdateDelay(), 10.0) * 1000));
	else
		has_event = SDL_PollEvent(&ev);

	while (has_event)
	{
		bool propagate_event = true;
		switch (ev.type)
		{
		case event_quit:
		{
			propagate_event = false;
			result = false;
		}
		break;
		case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
		{
			if (ev.window.windowID == SDL_GetWindowID(data->window))
			{
				propagate_event = false;
				result = false;
			}
		}
		break;
		case SDL_EVENT_WINDOW_RESIZED:
		case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
		{
			ConfigureSurface(*data);
		}
		break;
		case SDL_EVENT_WINDOW_FOCUS_LOST:
		{
			for (bool& suppressed : data->suppressed_key_down)
				suppressed = false;
		}
		break;
		case SDL_EVENT_KEY_UP:
		{
			const int scancode = static_cast<int>(ev.key.scancode);
			if (scancode >= 0 && scancode < SDL_SCANCODE_COUNT && data->suppressed_key_down[scancode])
			{
				data->suppressed_key_down[scancode] = false;
				propagate_event = false;
			}
		}
		break;
		case event_key_down:
		{
			const int scancode = static_cast<int>(ev.key.scancode);
			if (scancode >= 0 && scancode < SDL_SCANCODE_COUNT && data->suppressed_key_down[scancode])
			{
				propagate_event = false;
				suppress_text_input = true;
				break;
			}

			if (!ev.key.repeat && data->raw_key_down_callback &&
				data->raw_key_down_callback(static_cast<int>(ev.key.key), static_cast<int>(ev.key.scancode), static_cast<int>(ev.key.mod)))
			{
				if (scancode >= 0 && scancode < SDL_SCANCODE_COUNT)
					data->suppressed_key_down[scancode] = true;
				propagate_event = false;
				suppress_text_input = true;
				break;
			}

			suppress_text_input = false;
			propagate_event = false;
			const Rml::Input::KeyIdentifier key = RmlSDL::ConvertKey(GetKey(ev));
			const int key_modifier = RmlSDL::GetKeyModifierState();
			const float native_dp_ratio = GetDisplayScale();

			// See if we have any global shortcuts that take priority over the context.
			if (key_down_callback && !key_down_callback(context, key, key_modifier, native_dp_ratio, true))
				break;
			// Otherwise, hand the event over to the context by calling the input handler as normal.
			if (!RmlSDL::InputEventHandler(context, data->window, ev))
				break;
			// The key was not consumed by the context either, try keyboard shortcuts of lower priority.
			if (key_down_callback && !key_down_callback(context, key, key_modifier, native_dp_ratio, false))
				break;
		}
		break;
		case event_text_editing:
		{
			if (suppress_text_input)
			{
				propagate_event = false;
				break;
			}
			if (data->raw_text_editing_callback && data->raw_text_editing_callback(ev.edit.text))
			{
				propagate_event = false;
				break;
			}
			propagate_event = false;
			data->text_input_method_editor.HandleEdit(ev.edit);
		}
		break;
		default: break;
		}

		if (propagate_event && !(suppress_text_input && ev.type == SDL_EVENT_TEXT_INPUT))
			RmlSDL::InputEventHandler(context, data->window, ev);

		has_event = SDL_PollEvent(&ev);
	}

	// Keep the RmlUi context in sync with the drawable, but only when the window actually changed. This leaves
	// RmlUi in control of application-driven changes, such as its density-independent pixel ratio shortcuts.
	if (data->config.width > 0 && data->config.height > 0)
	{
		if (int(data->config.width) != data->synced_width || int(data->config.height) != data->synced_height)
		{
			data->synced_width = int(data->config.width);
			data->synced_height = int(data->config.height);
			context->SetDimensions({data->synced_width, data->synced_height});
		}
	}

	// The drawable is expressed in physical pixels, while layout sizes in RmlUi use density-independent pixels.
	const float pixel_density = SDL_GetWindowPixelDensity(data->window);
	if (pixel_density > 0.f && pixel_density != data->synced_pixel_density)
	{
		data->synced_pixel_density = pixel_density;
		context->SetDensityIndependentPixelRatio(pixel_density);
	}

	return result;
}

void Backend::SetRawKeyDownCallback(RawKeyDownCallback callback)
{
	RMLUI_ASSERT(data);
	data->raw_key_down_callback = std::move(callback);
}

void Backend::SetRawTextEditingCallback(RawTextEditingCallback callback)
{
	RMLUI_ASSERT(data);
	data->raw_text_editing_callback = std::move(callback);
}

void Backend::RequestExit()
{
	RMLUI_ASSERT(data);
	data->running = false;
}

void Backend::BeginFrame()
{
	RMLUI_ASSERT(data);

	data->frame_ready = false;

	int width = 0, height = 0;
	SDL_GetWindowSizeInPixels(data->window, &width, &height);
	if (width <= 0 || height <= 0)
		return;

	if (width != int(data->config.width) || height != int(data->config.height))
		ConfigureSurface(*data);

	WGPUSurfaceTexture surface_texture{};
	wgpuSurfaceGetCurrentTexture(data->surface, &surface_texture);

	const WGPUSurfaceGetCurrentTextureStatus status = surface_texture.status;
	if (status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal && status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
	{
		if (surface_texture.texture)
			wgpuTextureRelease(surface_texture.texture);

		if (static_cast<int>(status) == static_cast<int>(WGPUSurfaceGetCurrentTextureStatus_Occluded))
		{
			// The surface and swapchain remain valid while the window is not visible. Just skip the frame.
			return;
		}

		if (status == WGPUSurfaceGetCurrentTextureStatus_Timeout || status == WGPUSurfaceGetCurrentTextureStatus_Outdated ||
			status == WGPUSurfaceGetCurrentTextureStatus_Lost)
		{
			ConfigureSurface(*data);
			return;
		}

		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to acquire the WebGPU surface texture, status=%#.8x", static_cast<unsigned>(status));
		return;
	}

	WGPUTextureView surface_view = wgpuTextureCreateView(surface_texture.texture, nullptr);
	if (!surface_view)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create the WebGPU surface texture view.");
		wgpuTextureRelease(surface_texture.texture);
		return;
	}

	WGPUCommandEncoderDescriptor command_encoder_descriptor{};
	command_encoder_descriptor.label = {"rmlui_wgpu_frame_encoder", WGPU_STRLEN};
	WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(data->device, &command_encoder_descriptor);
	if (!command_encoder)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create the WebGPU command encoder.");
		wgpuTextureViewRelease(surface_view);
		wgpuTextureRelease(surface_texture.texture);
		return;
	}

	data->surface_texture = surface_texture;
	data->surface_view = surface_view;
	data->command_encoder = command_encoder;
	data->frame_ready = true;

	data->render_interface.BeginFrame(command_encoder, surface_view, data->config.width, data->config.height);
}

void Backend::PresentFrame()
{
	RMLUI_ASSERT(data);

	if (!data->frame_ready)
		return;

	data->render_interface.EndFrame();

	WGPUCommandBufferDescriptor command_buffer_descriptor{};
	command_buffer_descriptor.label = {"rmlui_wgpu_frame_commands", WGPU_STRLEN};
	WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(data->command_encoder, &command_buffer_descriptor);
	if (command_buffer)
	{
		WGPUCommandBuffer command_buffers[] = {command_buffer};
		wgpuQueueSubmit(data->queue, 1, command_buffers);
		const WGPUStatus present_status = wgpuSurfacePresent(data->surface);
		if (present_status != WGPUStatus_Success)
		{
			Rml::Log::Message(Rml::Log::LT_WARNING, "Failed to present the WebGPU surface, status=%#.8x", static_cast<unsigned>(present_status));
		}
		wgpuCommandBufferRelease(command_buffer);
	}
	else
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to finish the RmlUi WebGPU command buffer.");
	}

	wgpuCommandEncoderRelease(data->command_encoder);
	data->command_encoder = nullptr;
	wgpuTextureViewRelease(data->surface_view);
	data->surface_view = nullptr;
	wgpuTextureRelease(data->surface_texture.texture);
	data->surface_texture.texture = nullptr;
	data->frame_ready = false;
}
