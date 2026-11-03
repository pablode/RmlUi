#pragma once

#include <RmlUi/Core/RenderInterface.h>
#include <RmlUi/Core/Types.h>

#include <webgpu/webgpu.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#ifdef RMLUI_MATRIX_ROW_MAJOR
	#error "The SDL_WGPU backend requires RmlUi's column-major matrix layout (RMLUI_MATRIX_ROW_MAJOR is not supported)."
#endif

/**
    RmlUi render interface implemented on top of the WebGPU C API (wgpu-native).

    The renderer uses a single dynamic-offset storage buffer for per-draw data, combined texture samplers (repeat for
    images, clamp-to-edge for render targets), and a 1x1 white texture for untextured geometry. It implements the optional
    RmlUi features on top of that: clip masks through a stencil attachment, render layers through offscreen textures, image
    filters through fullscreen post-processing passes, and gradients/creation shaders.
 */
class RenderInterface_WGPU : public Rml::RenderInterface {
public:
	using BackgroundCallback = std::function<void(WGPUCommandEncoder command_encoder, WGPUTextureView surface_view, uint32_t width, uint32_t height)>;

	RenderInterface_WGPU(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat target_format);
	~RenderInterface_WGPU() override;

	void Shutdown();

	WGPUDevice GetDevice() const { return device; }
	WGPUQueue GetQueue() const { return queue; }
	WGPUTextureFormat GetTargetFormat() const { return target_format; }

	/// Sets an optional callback that renders a background into the swapchain surface before the UI is composited on top of it.
	/// The callback must clear the surface and finish any render pass it begins. Pass an empty callback to disable it.
	void SetBackgroundCallback(BackgroundCallback callback);

	/// Begins a frame by starting a render pass on an offscreen root layer. The renderer creates and manages the render passes, the
	/// backend only owns the command encoder.
	void BeginFrame(WGPUCommandEncoder command_encoder, WGPUTextureView surface_view, uint32_t width, uint32_t height);
	/// Ends the current render pass and composites the offscreen root layer to the swapchain surface. Does not submit the frame.
	void EndFrame();

	// -- Required functions --
	Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) override;
	void RenderGeometry(Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation, Rml::TextureHandle texture) override;
	void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) override;
	Rml::TextureHandle LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source) override;
	Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i source_dimensions) override;
	void ReleaseTexture(Rml::TextureHandle texture) override;
	void EnableScissorRegion(bool enable) override;
	void SetScissorRegion(Rml::Rectanglei region) override;

	// -- Optional functions --
	void EnableClipMask(bool enable) override;
	void RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation) override;
	void SetTransform(const Rml::Matrix4f* transform) override;
	Rml::LayerHandle PushLayer() override;
	void CompositeLayers(Rml::LayerHandle source, Rml::LayerHandle destination, Rml::BlendMode blend_mode,
		Rml::Span<const Rml::CompiledFilterHandle> filters) override;
	void PopLayer() override;
	Rml::TextureHandle SaveLayerAsTexture() override;
	Rml::CompiledFilterHandle SaveLayerAsMaskImage() override;
	Rml::CompiledFilterHandle CompileFilter(const Rml::String& name, const Rml::Dictionary& parameters) override;
	void ReleaseFilter(Rml::CompiledFilterHandle filter) override;
	Rml::CompiledShaderHandle CompileShader(const Rml::String& name, const Rml::Dictionary& parameters) override;
	void RenderShader(Rml::CompiledShaderHandle shader, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation,
		Rml::TextureHandle texture) override;
	void ReleaseShader(Rml::CompiledShaderHandle shader) override;

private:
	static constexpr uint32_t MAX_NUM_STOPS = 16;
	static constexpr size_t UNIFORM_STRIDE = 1024; // Dynamic offset alignment is at most 256, keep a multiple of it.
	static constexpr size_t MAX_DRAWS_PER_FRAME = 4096;
	static constexpr size_t UNIFORM_BUFFER_SIZE = UNIFORM_STRIDE * MAX_DRAWS_PER_FRAME;

	struct alignas(16) DrawUniforms {
		Rml::Matrix4f transform;
		Rml::Vector2f translation;
		Rml::Vector2f texel_offset;
		Rml::Vector2f p;
		Rml::Vector2f v;
		Rml::Vector4f color;
		Rml::Matrix4f color_matrix;
		Rml::Vector4f blur_weights;
		Rml::Vector4f stop_colors[MAX_NUM_STOPS];
		float stop_positions[MAX_NUM_STOPS];
		uint32_t func;
		uint32_t num_stops;
		float opacity;
		float padding;
		Rml::Vector2f tex_coord_min;
		Rml::Vector2f tex_coord_max;
		Rml::Vector2f dimensions;
		float time;
	};

	static constexpr size_t UNIFORM_STRUCT_SIZE = sizeof(DrawUniforms);
	static_assert(sizeof(DrawUniforms) == 560, "DrawUniforms must match the WGSL storage buffer layout.");
	static_assert(offsetof(DrawUniforms, color) == 96, "DrawUniforms layout mismatch.");
	static_assert(offsetof(DrawUniforms, color_matrix) == 112, "DrawUniforms layout mismatch.");
	static_assert(offsetof(DrawUniforms, blur_weights) == 176, "DrawUniforms layout mismatch.");
	static_assert(offsetof(DrawUniforms, stop_colors) == 192, "DrawUniforms layout mismatch.");
	static_assert(offsetof(DrawUniforms, stop_positions) == 448, "DrawUniforms layout mismatch.");
	static_assert(offsetof(DrawUniforms, func) == 512, "DrawUniforms layout mismatch.");
	static_assert(offsetof(DrawUniforms, tex_coord_min) == 528, "DrawUniforms layout mismatch.");

	struct Geometry {
		WGPUBuffer vertex_buffer = nullptr;
		WGPUBuffer index_buffer = nullptr;
		uint64_t vertex_buffer_size = 0;
		uint64_t index_buffer_size = 0;
		uint32_t index_count = 0;
	};

	struct Texture {
		WGPUTexture texture = nullptr;
		WGPUTextureView view = nullptr;
		WGPUBindGroup bind_group = nullptr;
	};

	struct RenderTarget {
		WGPUTexture texture = nullptr;
		WGPUTextureView view = nullptr;
		WGPUBindGroup bind_group = nullptr;
		uint32_t width = 0;
		uint32_t height = 0;
	};

	enum class FilterType { Invalid, Passthrough, Blur, DropShadow, ColorMatrix, MaskImage };

	struct CompiledFilter {
		FilterType type = FilterType::Invalid;
		float blend_factor = 1.f;
		float sigma = 0.f;
		Rml::Vector2f offset;
		Rml::Vector4f color;
		Rml::Matrix4f color_matrix;
	};

	enum class ShaderType { Invalid, LinearGradient, RadialGradient, ConicGradient, RepeatingLinearGradient, RepeatingRadialGradient, RepeatingConicGradient, Creation };

	struct CompiledShader {
		ShaderType type = ShaderType::Invalid;
		uint32_t func = 0;
		Rml::Vector2f p;
		Rml::Vector2f v;
		uint32_t num_stops = 0;
		Rml::Vector4f stop_colors[MAX_NUM_STOPS]{};
		float stop_positions[MAX_NUM_STOPS]{};
		Rml::Vector2f dimensions;
	};

	void CreatePipelines();
	void CreateWhiteTexture();

	Texture* CreateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i dimensions);
	Texture* CreateTextureObject(WGPUTexture texture, Rml::Vector2i dimensions);
	WGPUBindGroup CreateBindGroup(WGPUTextureView tex0, WGPUTextureView tex1, WGPUSampler texture_sampler);

	RenderTarget CreateRenderTarget(WGPUTextureUsage extra_usage);
	void ReleaseRenderTarget(RenderTarget& target);
	void EnsureTargetStorage();

	void BeginPass(RenderTarget* target, bool clear_color, WGPUColor clear_value, bool clear_stencil, uint32_t stencil_value);
	void EndPass();
	RenderTarget* GetTarget(Rml::LayerHandle handle);

	void SetScissor();
	bool WriteUniform(const DrawUniforms& uniforms, uint32_t& out_offset);
	void DrawGeometryWithPipeline(WGPURenderPipeline pipeline, Geometry* geometry, Rml::Vector2f translation, WGPUBindGroup bind_group, bool clipped);
	void DrawFullscreenWithPipeline(WGPURenderPipeline pipeline, WGPUBindGroup bind_group, const DrawUniforms& uniforms);
	WGPURenderPipeline ChooseGeometryPipeline(WGPURenderPipeline normal, WGPURenderPipeline clipped) const;
	void ApplyFilter(const CompiledFilter& filter);
	void ApplyFilters(Rml::Span<const Rml::CompiledFilterHandle> filters);
	void ReleasePendingResources();

	WGPUDevice device = nullptr;
	WGPUQueue queue = nullptr;
	WGPUTextureFormat target_format = WGPUTextureFormat_Undefined;

	BackgroundCallback background_callback;

	WGPUCommandEncoder command_encoder = nullptr;
	WGPURenderPassEncoder render_pass = nullptr;

	RenderTarget surface_target;
	RenderTarget root_target;
	RenderTarget* current_target = nullptr;
	std::vector<RenderTarget> layers;
	RenderTarget postprocess_primary;
	RenderTarget postprocess_secondary;
	RenderTarget postprocess_tertiary;
	RenderTarget blend_mask;

	WGPUTexture stencil_texture = nullptr;
	WGPUTextureView stencil_view = nullptr;

	uint32_t viewport_width = 0;
	uint32_t viewport_height = 0;
	uint32_t allocated_width = 0;
	uint32_t allocated_height = 0;
	size_t layer_depth = 0;

	uint64_t uniform_offset = 0;
	Rml::Matrix4f projection;
	Rml::Matrix4f transform;
	Rml::Rectanglei scissor_region;
	bool scissor_enabled = false;
	bool uniform_overflow_reported = false;

	bool clip_mask_enabled = false;
	uint32_t clip_mask_ref = 1;

	WGPUBindGroupLayout bind_group_layout = nullptr;
	WGPUPipelineLayout pipeline_layout = nullptr;
	WGPUSampler sampler = nullptr;
	WGPUSampler render_target_sampler = nullptr;
	WGPUBuffer uniform_buffer = nullptr;

	WGPURenderPipeline pipeline_main = nullptr;
	WGPURenderPipeline pipeline_main_clipped = nullptr;
	WGPURenderPipeline pipeline_mask_replace = nullptr;
	WGPURenderPipeline pipeline_mask_increment = nullptr;
	WGPURenderPipeline pipeline_gradient = nullptr;
	WGPURenderPipeline pipeline_gradient_clipped = nullptr;
	WGPURenderPipeline pipeline_creation = nullptr;
	WGPURenderPipeline pipeline_creation_clipped = nullptr;
	WGPURenderPipeline pipeline_passthrough_blend = nullptr;
	WGPURenderPipeline pipeline_passthrough_replace = nullptr;
	WGPURenderPipeline pipeline_opacity = nullptr;
	WGPURenderPipeline pipeline_blur = nullptr;
	WGPURenderPipeline pipeline_drop_shadow = nullptr;
	WGPURenderPipeline pipeline_color_matrix = nullptr;
	WGPURenderPipeline pipeline_blend_mask = nullptr;

	Texture* white_texture = nullptr;
	Rml::CompiledGeometryHandle fullscreen_quad = 0;

	std::vector<WGPUBuffer> pending_release_buffers;
	std::vector<WGPUTexture> pending_release_textures;
	std::vector<WGPUTextureView> pending_release_texture_views;
	std::vector<WGPUBindGroup> pending_release_bind_groups;
};
