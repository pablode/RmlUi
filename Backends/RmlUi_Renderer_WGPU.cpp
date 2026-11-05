#include "RmlUi_Renderer_WGPU.h"

#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/DecorationTypes.h>
#include <RmlUi/Core/Dictionary.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include <RmlUi/Core/Math.h>
#include <RmlUi/Core/SystemInterface.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr WGPUColor clear_color = {0.1, 0.1, 0.12, 1.0};
constexpr WGPUColor transparent_clear = {0.0, 0.0, 0.0, 0.0};

constexpr const char* shader_common = R"(
struct DrawUniforms {
	transform: mat4x4<f32>,
	translation: vec2<f32>,
	texel_offset: vec2<f32>,
	p: vec2<f32>,
	v: vec2<f32>,
	color: vec4<f32>,
	color_matrix: mat4x4<f32>,
	blur_weights: vec4<f32>,
	stop_colors: array<vec4<f32>, 16>,
	stop_positions: array<f32, 16>,
	func: u32,
	num_stops: u32,
	opacity: f32,
	padding: f32,
	tex_coord_min: vec2<f32>,
	tex_coord_max: vec2<f32>,
	dimensions: vec2<f32>,
	time: f32,
};

@group(0) @binding(0) var<storage, read> u: DrawUniforms;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var tex0: texture_2d<f32>;
@group(0) @binding(3) var tex1: texture_2d<f32>;

struct VertexInput {
	@location(0) position: vec2<f32>,
	@location(1) color: vec4<f32>,
	@location(2) tex_coord: vec2<f32>,
};

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) color: vec4<f32>,
	@location(1) tex_coord: vec2<f32>,
};

struct FullscreenOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) tex_coord: vec2<f32>,
};
)";

constexpr const char* shader_vs_main = R"(
@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
	var output: VertexOutput;
	var clip = u.transform * vec4<f32>(input.position + u.translation, 0.0, 1.0);
	// RmlUi uses OpenGL-style clip coordinates with z in [-w, w]. WebGPU expects z in [0, w].
	clip.z = 0.5 * (clip.z + clip.w);
	output.position = clip;
	output.color = input.color;
	output.tex_coord = input.tex_coord;
	return output;
}
)";

constexpr const char* shader_vs_fullscreen = R"(
@vertex
fn vs_fullscreen(input: VertexInput) -> FullscreenOutput {
	var output: FullscreenOutput;
	output.position = vec4<f32>(input.position, 0.0, 1.0);
	output.tex_coord = input.tex_coord;
	return output;
}
)";

constexpr const char* shader_fs_main = R"(
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
	return input.color * textureSample(tex0, linear_sampler, input.tex_coord);
}
)";

constexpr const char* shader_fs_passthrough = R"(
@fragment
fn fs_passthrough(input: FullscreenOutput) -> @location(0) vec4<f32> {
	return textureSample(tex0, linear_sampler, input.tex_coord);
}
)";

constexpr const char* shader_fs_opacity = R"(
@fragment
fn fs_opacity(input: FullscreenOutput) -> @location(0) vec4<f32> {
	return textureSample(tex0, linear_sampler, input.tex_coord) * u.opacity;
}
)";

constexpr const char* shader_fs_gradient = R"(
@fragment
fn fs_gradient(input: VertexOutput) -> @location(0) vec4<f32> {
	var t = 0.0;
	let func = u.func;
	let pos = input.tex_coord;
	if (func == 0u || func == 3u) {
		let dist_square = dot(u.v, u.v);
		let v = pos - u.p;
		t = dot(u.v, v) / dist_square;
	} else if (func == 1u || func == 4u) {
		let v = pos - u.p;
		t = length(u.v * v);
	} else if (func == 2u || func == 5u) {
		let r = mat2x2<f32>(u.v.x, -u.v.y, u.v.y, u.v.x);
		let v = r * (pos - u.p);
		t = 0.5 + atan2(-v.x, v.y) / (2.0 * 3.14159265359);
	}

	if (func == 3u || func == 4u || func == 5u) {
		let t0 = u.stop_positions[0];
		let t1 = u.stop_positions[u.num_stops - 1u];
		let delta = t1 - t0;
		t = t0 + (t - t0) - delta * floor((t - t0) / delta);
	}

	var color = u.stop_colors[0];
	for (var i = 1u; i < u.num_stops; i = i + 1u) {
		let s = smoothstep(u.stop_positions[i - 1u], u.stop_positions[i], t);
		color = mix(color, u.stop_colors[i], s);
	}
	return input.color * color;
}
)";

constexpr const char* shader_fs_creation = R"(
@fragment
fn fs_creation(input: VertexOutput) -> @location(0) vec4<f32> {
	let t = u.time;
	var c: array<f32, 3>;
	var l = 0.0;
	for (var i = 0; i < 3; i = i + 1) {
		var p = input.tex_coord;
		var uv = p;
		p = p - vec2<f32>(0.5);
		p.x = p.x * (u.dimensions.x / u.dimensions.y);
		let z = t + f32(i) * 0.07;
		l = length(p);
		uv = uv + p / l * (sin(z) + 1.0) * abs(sin(l * 9.0 - z - z));
		let wrapped = uv - floor(uv);
		c[i] = 0.01 / length(wrapped - vec2<f32>(0.5));
	}
	return vec4<f32>(c[0] / l, c[1] / l, c[2] / l, input.color.a);
}
)";

constexpr const char* shader_fs_blur = R"(
fn sample_region(uv: vec2<f32>) -> vec4<f32> {
	let in_region = step(u.tex_coord_min, uv) * step(uv, u.tex_coord_max);
	return textureSample(tex0, linear_sampler, uv) * in_region.x * in_region.y;
}

@fragment
fn fs_blur(input: FullscreenOutput) -> @location(0) vec4<f32> {
	let step_uv = u.texel_offset;
	let w = u.blur_weights;
	var color = sample_region(input.tex_coord) * w.x;
	color += (sample_region(input.tex_coord + step_uv) + sample_region(input.tex_coord - step_uv)) * w.y;
	color += (sample_region(input.tex_coord + step_uv * 2.0) + sample_region(input.tex_coord - step_uv * 2.0)) * w.z;
	color += (sample_region(input.tex_coord + step_uv * 3.0) + sample_region(input.tex_coord - step_uv * 3.0)) * w.w;
	return color;
}
)";

constexpr const char* shader_fs_drop_shadow = R"(
@fragment
fn fs_drop_shadow(input: FullscreenOutput) -> @location(0) vec4<f32> {
	let uv = input.tex_coord + u.texel_offset;
	let in_region = step(u.tex_coord_min, uv) * step(uv, u.tex_coord_max);
	let alpha = textureSample(tex0, linear_sampler, uv).a * in_region.x * in_region.y;
	return u.color * alpha;
}
)";

constexpr const char* shader_fs_color_matrix = R"(
@fragment
fn fs_color_matrix(input: FullscreenOutput) -> @location(0) vec4<f32> {
	let tex_color = textureSample(tex0, linear_sampler, input.tex_coord);
	let transformed = u.color_matrix * tex_color;
	return vec4<f32>(transformed.rgb, tex_color.a);
}
)";

constexpr const char* shader_fs_blend_mask = R"(
@fragment
fn fs_blend_mask(input: FullscreenOutput) -> @location(0) vec4<f32> {
	let tex_color = textureSample(tex0, linear_sampler, input.tex_coord);
	let mask_alpha = textureSample(tex1, linear_sampler, input.tex_coord).a;
	return tex_color * mask_alpha;
}
)";

WGPUStringView MakeStringView(const char* str)
{
	return WGPUStringView{str, WGPU_STRLEN};
}

uint32_t AlignTo(uint32_t value, uint32_t alignment)
{
	return (value + alignment - 1) / alignment * alignment;
}

/**
	Decodes a TGA image into top-left originated, premultiplied RGBA pixels.
	Supports uncompressed and RLE true-color images with 24 or 32 bits per pixel.
 */
bool DecodeTGA(const Rml::byte* data, size_t size, Rml::Vector2i& dimensions, std::vector<Rml::byte>& pixels)
{
	if (size < 18)
		return false;

	const Rml::byte* header = data;
	const Rml::byte id_length = header[0];
	const Rml::byte color_map_type = header[1];
	const Rml::byte image_type = header[2];
	const uint16_t width = static_cast<uint16_t>(header[12] | (header[13] << 8));
	const uint16_t height = static_cast<uint16_t>(header[14] | (header[15] << 8));
	const Rml::byte pixel_depth = header[16];
	const Rml::byte descriptor = header[17];

	if (color_map_type != 0 || (image_type != 2 && image_type != 10))
		return false;
	if (pixel_depth != 24 && pixel_depth != 32)
		return false;
	if (width == 0 || height == 0)
		return false;

	const size_t bytes_per_pixel = pixel_depth / 8;
	const size_t pixel_count = size_t(width) * size_t(height);
	pixels.resize(pixel_count * 4);

	size_t read_offset = 18 + id_length;
	size_t write_offset = 0;

	auto write_pixel = [&](const Rml::byte* source) {
		const Rml::byte blue = source[0];
		const Rml::byte green = source[1];
		const Rml::byte red = source[2];
		const Rml::byte alpha = (bytes_per_pixel == 4) ? source[3] : Rml::byte(255);
		pixels[write_offset + 0] = Rml::byte(int(red) * int(alpha) / 255);
		pixels[write_offset + 1] = Rml::byte(int(green) * int(alpha) / 255);
		pixels[write_offset + 2] = Rml::byte(int(blue) * int(alpha) / 255);
		pixels[write_offset + 3] = alpha;
		write_offset += 4;
	};

	if (image_type == 2)
	{
		if (read_offset + pixel_count * bytes_per_pixel > size)
			return false;

		for (size_t i = 0; i < pixel_count; i++)
		{
			write_pixel(data + read_offset);
			read_offset += bytes_per_pixel;
		}
	}
	else
	{
		while (write_offset < pixels.size())
		{
			if (read_offset >= size)
				return false;

			const Rml::byte packet = data[read_offset++];
			const size_t count = size_t(packet & 0x7F) + 1;
			if (count > (pixel_count - write_offset / 4))
				return false;

			if (packet & 0x80)
			{
				if (read_offset + bytes_per_pixel > size)
					return false;
				const Rml::byte* pixel = data + read_offset;
				read_offset += bytes_per_pixel;
				for (size_t i = 0; i < count; i++)
					write_pixel(pixel);
			}
			else
			{
				if (read_offset + count * bytes_per_pixel > size)
					return false;
				for (size_t i = 0; i < count; i++)
				{
					write_pixel(data + read_offset);
					read_offset += bytes_per_pixel;
				}
			}
		}
	}

	// TGA images are bottom-left originated unless the top-left origin bit is set.
	if ((descriptor & 0x20) == 0)
	{
		const size_t row_size = size_t(width) * 4;
		std::vector<Rml::byte> row(row_size);
		for (size_t y = 0; y < size_t(height) / 2; y++)
		{
			Rml::byte* top = pixels.data() + y * row_size;
			Rml::byte* bottom = pixels.data() + (size_t(height) - 1 - y) * row_size;
			std::memcpy(row.data(), top, row_size);
			std::memcpy(top, bottom, row_size);
			std::memcpy(bottom, row.data(), row_size);
		}
	}

	dimensions = {int(width), int(height)};
	return true;
}

WGPUShaderModule CreateShaderModule(WGPUDevice device, const std::string& source)
{
	WGPUShaderSourceWGSL wgsl_source{};
	wgsl_source.chain.sType = WGPUSType_ShaderSourceWGSL;
	wgsl_source.code = WGPUStringView{source.c_str(), WGPU_STRLEN};

	WGPUShaderModuleDescriptor descriptor{};
	descriptor.label = MakeStringView("rmlui_wgpu_shader");
	descriptor.nextInChain = &wgsl_source.chain;
	return wgpuDeviceCreateShaderModule(device, &descriptor);
}

WGPURenderPipeline CreatePipeline(WGPUDevice device, WGPUPipelineLayout pipeline_layout, WGPUTextureFormat target_format, WGPUShaderModule module,
	const char* vertex_entry, const char* fragment_entry, bool blend_enabled, WGPUColorWriteMask write_mask, WGPUCompareFunction stencil_compare,
	WGPUStencilOperation stencil_pass_op)
{
	WGPUVertexAttribute attributes[3]{};
	attributes[0].format = WGPUVertexFormat_Float32x2;
	attributes[0].offset = offsetof(Rml::Vertex, position);
	attributes[0].shaderLocation = 0;
	attributes[1].format = WGPUVertexFormat_Unorm8x4;
	attributes[1].offset = offsetof(Rml::Vertex, colour);
	attributes[1].shaderLocation = 1;
	attributes[2].format = WGPUVertexFormat_Float32x2;
	attributes[2].offset = offsetof(Rml::Vertex, tex_coord);
	attributes[2].shaderLocation = 2;

	WGPUVertexBufferLayout vertex_buffer_layout{};
	vertex_buffer_layout.stepMode = WGPUVertexStepMode_Vertex;
	vertex_buffer_layout.arrayStride = sizeof(Rml::Vertex);
	vertex_buffer_layout.attributeCount = 3;
	vertex_buffer_layout.attributes = attributes;

	WGPUVertexState vertex_state{};
	vertex_state.module = module;
	vertex_state.entryPoint = MakeStringView(vertex_entry);
	vertex_state.bufferCount = 1;
	vertex_state.buffers = &vertex_buffer_layout;

	WGPUBlendComponent color_blend{};
	color_blend.operation = WGPUBlendOperation_Add;
	color_blend.srcFactor = WGPUBlendFactor_One;
	color_blend.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
	WGPUBlendComponent alpha_blend = color_blend;

	WGPUBlendState blend_state{};
	blend_state.color = color_blend;
	blend_state.alpha = alpha_blend;

	WGPUColorTargetState color_target{};
	color_target.format = target_format;
	color_target.blend = blend_enabled ? &blend_state : nullptr;
	color_target.writeMask = write_mask;

	WGPUFragmentState fragment_state{};
	fragment_state.module = module;
	fragment_state.entryPoint = MakeStringView(fragment_entry);
	fragment_state.targetCount = 1;
	fragment_state.targets = &color_target;

	WGPUPrimitiveState primitive_state{};
	primitive_state.topology = WGPUPrimitiveTopology_TriangleList;
	primitive_state.frontFace = WGPUFrontFace_CCW;
	primitive_state.cullMode = WGPUCullMode_None;

	WGPUStencilFaceState stencil_face{};
	stencil_face.compare = stencil_compare;
	stencil_face.failOp = WGPUStencilOperation_Keep;
	stencil_face.depthFailOp = WGPUStencilOperation_Keep;
	stencil_face.passOp = stencil_pass_op;

	WGPUDepthStencilState depth_stencil_state{};
	depth_stencil_state.format = WGPUTextureFormat_Depth24PlusStencil8;
	depth_stencil_state.depthWriteEnabled = WGPUOptionalBool_False;
	depth_stencil_state.depthCompare = WGPUCompareFunction_Always;
	depth_stencil_state.stencilFront = stencil_face;
	depth_stencil_state.stencilBack = stencil_face;
	depth_stencil_state.stencilReadMask = 0xFF;
	depth_stencil_state.stencilWriteMask = 0xFF;

	WGPUMultisampleState multisample_state{};
	multisample_state.count = 1;
	multisample_state.mask = 0xFFFFFFFF;

	WGPURenderPipelineDescriptor pipeline_descriptor{};
	pipeline_descriptor.label = MakeStringView("rmlui_wgpu_pipeline");
	pipeline_descriptor.layout = pipeline_layout;
	pipeline_descriptor.vertex = vertex_state;
	pipeline_descriptor.primitive = primitive_state;
	pipeline_descriptor.depthStencil = &depth_stencil_state;
	pipeline_descriptor.multisample = multisample_state;
	pipeline_descriptor.fragment = &fragment_state;
	return wgpuDeviceCreateRenderPipeline(device, &pipeline_descriptor);
}

} // namespace

namespace {

void UploadTexture(WGPUQueue queue, WGPUTexture texture, uint32_t width, uint32_t height, const Rml::byte* data, size_t size)
{
	const uint32_t row_bytes = width * 4;
	const uint32_t padded_row_bytes = AlignTo(row_bytes, 256);

	std::vector<Rml::byte> padded_data;
	const Rml::byte* upload_data = data;
	size_t upload_size = size;

	if (padded_row_bytes != row_bytes)
	{
		padded_data.resize(size_t(padded_row_bytes) * height);
		for (uint32_t y = 0; y < height; y++)
			std::memcpy(padded_data.data() + size_t(y) * padded_row_bytes, data + size_t(y) * row_bytes, row_bytes);
		upload_data = padded_data.data();
		upload_size = padded_data.size();
	}

	WGPUTexelCopyTextureInfo destination{};
	destination.texture = texture;
	destination.mipLevel = 0;
	destination.origin = {0, 0, 0};
	destination.aspect = WGPUTextureAspect_All;

	WGPUTexelCopyBufferLayout data_layout{};
	data_layout.offset = 0;
	data_layout.bytesPerRow = padded_row_bytes;
	data_layout.rowsPerImage = height;

	WGPUExtent3D write_size = {width, height, 1};
	wgpuQueueWriteTexture(queue, &destination, upload_data, upload_size, &data_layout, &write_size);
}

} // namespace

RenderInterface_WGPU::RenderInterface_WGPU(WGPUDevice device, WGPUQueue queue, WGPUTextureFormat target_format) :
	device(device), queue(queue), target_format(target_format), projection(Rml::Matrix4f::Identity()), transform(Rml::Matrix4f::Identity())
{
	CreatePipelines();
	CreateWhiteTexture();

	// Unit quad in normalized device coordinates, used for fullscreen post-processing passes.
	const Rml::ColourbPremultiplied white(255, 255, 255, 255);
	std::vector<Rml::Vertex> vertices = {
		{{-1.f, 1.f}, white, {0.f, 0.f}},
		{{1.f, 1.f}, white, {1.f, 0.f}},
		{{1.f, -1.f}, white, {1.f, 1.f}},
		{{-1.f, -1.f}, white, {0.f, 1.f}},
	};
	std::vector<int> indices = {0, 1, 2, 0, 2, 3};
	fullscreen_quad = CompileGeometry(Rml::Span<const Rml::Vertex>(vertices), Rml::Span<const int>(indices));
}

RenderInterface_WGPU::~RenderInterface_WGPU()
{
	Shutdown();
}

void RenderInterface_WGPU::CreatePipelines()
{
	WGPUBindGroupLayoutEntry entries[4]{};
	entries[0].binding = 0;
	entries[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
	entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
	entries[0].buffer.hasDynamicOffset = true;
	entries[0].buffer.minBindingSize = UNIFORM_STRUCT_SIZE;
	entries[1].binding = 1;
	entries[1].visibility = WGPUShaderStage_Fragment;
	entries[1].sampler.type = WGPUSamplerBindingType_Filtering;
	entries[2].binding = 2;
	entries[2].visibility = WGPUShaderStage_Fragment;
	entries[2].texture.sampleType = WGPUTextureSampleType_Float;
	entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;
	entries[3].binding = 3;
	entries[3].visibility = WGPUShaderStage_Fragment;
	entries[3].texture.sampleType = WGPUTextureSampleType_Float;
	entries[3].texture.viewDimension = WGPUTextureViewDimension_2D;

	WGPUBindGroupLayoutDescriptor bind_group_layout_descriptor{};
	bind_group_layout_descriptor.label = MakeStringView("rmlui_wgpu_bind_group_layout");
	bind_group_layout_descriptor.entryCount = 4;
	bind_group_layout_descriptor.entries = entries;
	bind_group_layout = wgpuDeviceCreateBindGroupLayout(device, &bind_group_layout_descriptor);

	WGPUPipelineLayoutDescriptor pipeline_layout_descriptor{};
	pipeline_layout_descriptor.label = MakeStringView("rmlui_wgpu_pipeline_layout");
	pipeline_layout_descriptor.bindGroupLayoutCount = 1;
	pipeline_layout_descriptor.bindGroupLayouts = &bind_group_layout;
	pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_descriptor);

	WGPUSamplerDescriptor sampler_descriptor{};
	sampler_descriptor.label = MakeStringView("rmlui_wgpu_sampler");
	sampler_descriptor.addressModeU = WGPUAddressMode_Repeat;
	sampler_descriptor.addressModeV = WGPUAddressMode_Repeat;
	sampler_descriptor.addressModeW = WGPUAddressMode_Repeat;
	sampler_descriptor.magFilter = WGPUFilterMode_Linear;
	sampler_descriptor.minFilter = WGPUFilterMode_Linear;
	sampler_descriptor.mipmapFilter = WGPUMipmapFilterMode_Linear;
	sampler_descriptor.lodMinClamp = 0.f;
	sampler_descriptor.lodMaxClamp = 1.f;
	sampler_descriptor.maxAnisotropy = 1;
	sampler = wgpuDeviceCreateSampler(device, &sampler_descriptor);

	WGPUSamplerDescriptor render_target_sampler_descriptor = sampler_descriptor;
	render_target_sampler_descriptor.label = MakeStringView("rmlui_wgpu_render_target_sampler");
	render_target_sampler_descriptor.addressModeU = WGPUAddressMode_ClampToEdge;
	render_target_sampler_descriptor.addressModeV = WGPUAddressMode_ClampToEdge;
	render_target_sampler_descriptor.addressModeW = WGPUAddressMode_ClampToEdge;
	render_target_sampler = wgpuDeviceCreateSampler(device, &render_target_sampler_descriptor);

	WGPUBufferDescriptor uniform_buffer_descriptor{};
	uniform_buffer_descriptor.label = MakeStringView("rmlui_wgpu_uniforms");
	uniform_buffer_descriptor.size = UNIFORM_BUFFER_SIZE;
	uniform_buffer_descriptor.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
	uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_buffer_descriptor);

	if (!bind_group_layout || !pipeline_layout || !sampler || !render_target_sampler || !uniform_buffer)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create the RmlUi WebGPU shared resources.");
		return;
	}

	auto make_module = [this](const char* vertex_entry, const char* fragment_source) {
		std::string source = shader_common;
		source += (std::strcmp(vertex_entry, "vs_fullscreen") == 0) ? shader_vs_fullscreen : shader_vs_main;
		source += fragment_source;
		return CreateShaderModule(device, source);
	};

	struct PipelineDefinition {
		WGPURenderPipeline* target;
		const char* fragment_source;
		const char* vertex_entry;
		const char* fragment_entry;
		bool blend;
		WGPUColorWriteMask write_mask;
		WGPUCompareFunction stencil_compare;
		WGPUStencilOperation stencil_pass_op;
	};

	const PipelineDefinition definitions[] = {
		{&pipeline_main, shader_fs_main, "vs_main", "fs_main", true, WGPUColorWriteMask_All, WGPUCompareFunction_Always, WGPUStencilOperation_Keep},
		{&pipeline_main_clipped, shader_fs_main, "vs_main", "fs_main", true, WGPUColorWriteMask_All, WGPUCompareFunction_Equal, WGPUStencilOperation_Keep},
		{&pipeline_mask_replace, shader_fs_main, "vs_main", "fs_main", false, WGPUColorWriteMask_None, WGPUCompareFunction_Always, WGPUStencilOperation_Replace},
		{&pipeline_mask_increment, shader_fs_main, "vs_main", "fs_main", false, WGPUColorWriteMask_None, WGPUCompareFunction_Always, WGPUStencilOperation_IncrementClamp},
		{&pipeline_gradient, shader_fs_gradient, "vs_main", "fs_gradient", true, WGPUColorWriteMask_All, WGPUCompareFunction_Always, WGPUStencilOperation_Keep},
		{&pipeline_gradient_clipped, shader_fs_gradient, "vs_main", "fs_gradient", true, WGPUColorWriteMask_All, WGPUCompareFunction_Equal, WGPUStencilOperation_Keep},
		{&pipeline_creation, shader_fs_creation, "vs_main", "fs_creation", true, WGPUColorWriteMask_All, WGPUCompareFunction_Always, WGPUStencilOperation_Keep},
		{&pipeline_creation_clipped, shader_fs_creation, "vs_main", "fs_creation", true, WGPUColorWriteMask_All, WGPUCompareFunction_Equal, WGPUStencilOperation_Keep},
		{&pipeline_passthrough_blend, shader_fs_passthrough, "vs_fullscreen", "fs_passthrough", true, WGPUColorWriteMask_All, WGPUCompareFunction_Always, WGPUStencilOperation_Keep},
		{&pipeline_passthrough_replace, shader_fs_passthrough, "vs_fullscreen", "fs_passthrough", false, WGPUColorWriteMask_All, WGPUCompareFunction_Always, WGPUStencilOperation_Keep},
		{&pipeline_opacity, shader_fs_opacity, "vs_fullscreen", "fs_opacity", false, WGPUColorWriteMask_All, WGPUCompareFunction_Always, WGPUStencilOperation_Keep},
		{&pipeline_blur, shader_fs_blur, "vs_fullscreen", "fs_blur", false, WGPUColorWriteMask_All, WGPUCompareFunction_Always, WGPUStencilOperation_Keep},
		{&pipeline_drop_shadow, shader_fs_drop_shadow, "vs_fullscreen", "fs_drop_shadow", false, WGPUColorWriteMask_All, WGPUCompareFunction_Always, WGPUStencilOperation_Keep},
		{&pipeline_color_matrix, shader_fs_color_matrix, "vs_fullscreen", "fs_color_matrix", false, WGPUColorWriteMask_All, WGPUCompareFunction_Always, WGPUStencilOperation_Keep},
		{&pipeline_blend_mask, shader_fs_blend_mask, "vs_fullscreen", "fs_blend_mask", false, WGPUColorWriteMask_All, WGPUCompareFunction_Always, WGPUStencilOperation_Keep},
	};

	for (const PipelineDefinition& definition : definitions)
	{
		WGPUShaderModule module = make_module(definition.vertex_entry, definition.fragment_source);
		if (!module)
			continue;
		*definition.target = CreatePipeline(device, pipeline_layout, target_format, module, definition.vertex_entry, definition.fragment_entry,
			definition.blend, definition.write_mask, definition.stencil_compare, definition.stencil_pass_op);
		wgpuShaderModuleRelease(module);
		if (!*definition.target)
		{
			Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create an RmlUi WebGPU pipeline.");
		}
	}
}

WGPUBindGroup RenderInterface_WGPU::CreateBindGroup(WGPUTextureView tex0, WGPUTextureView tex1, WGPUSampler texture_sampler)
{
	if (!bind_group_layout || !uniform_buffer || !texture_sampler || !tex0 || !tex1)
		return nullptr;

	WGPUBindGroupEntry entries[4]{};
	entries[0].binding = 0;
	entries[0].buffer = uniform_buffer;
	entries[0].offset = 0;
	entries[0].size = UNIFORM_STRUCT_SIZE;
	entries[1].binding = 1;
	entries[1].sampler = texture_sampler;
	entries[2].binding = 2;
	entries[2].textureView = tex0;
	entries[3].binding = 3;
	entries[3].textureView = tex1;

	WGPUBindGroupDescriptor descriptor{};
	descriptor.label = MakeStringView("rmlui_wgpu_bind_group");
	descriptor.layout = bind_group_layout;
	descriptor.entryCount = 4;
	descriptor.entries = entries;
	return wgpuDeviceCreateBindGroup(device, &descriptor);
}

RenderInterface_WGPU::Texture* RenderInterface_WGPU::CreateTextureObject(WGPUTexture texture, Rml::Vector2i dimensions)
{
	Texture* result = new Texture();
	result->texture = texture;
	result->view = wgpuTextureCreateView(texture, nullptr);
	if (!result->view)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create a RmlUi WebGPU texture view.");
		wgpuTextureRelease(texture);
		delete result;
		return nullptr;
	}

	WGPUTextureView tex1 = white_texture ? white_texture->view : result->view;
	result->bind_group = CreateBindGroup(result->view, tex1, sampler);
	if (!result->bind_group)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create a RmlUi WebGPU texture bind group.");
		wgpuTextureViewRelease(result->view);
		wgpuTextureRelease(texture);
		delete result;
		return nullptr;
	}

	(void) dimensions;
	return result;
}

RenderInterface_WGPU::Texture* RenderInterface_WGPU::CreateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i dimensions)
{
	if (!device || !queue || !bind_group_layout || !uniform_buffer || !sampler || dimensions.x <= 0 || dimensions.y <= 0)
		return nullptr;

	const uint32_t width = static_cast<uint32_t>(dimensions.x);
	const uint32_t height = static_cast<uint32_t>(dimensions.y);
	if (source.size() < size_t(width) * height * 4)
		return nullptr;

	WGPUTextureDescriptor texture_descriptor{};
	texture_descriptor.label = MakeStringView("rmlui_wgpu_texture");
	texture_descriptor.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
	texture_descriptor.dimension = WGPUTextureDimension_2D;
	texture_descriptor.size = {width, height, 1};
	texture_descriptor.format = WGPUTextureFormat_RGBA8Unorm;
	texture_descriptor.mipLevelCount = 1;
	texture_descriptor.sampleCount = 1;

	WGPUTexture texture = wgpuDeviceCreateTexture(device, &texture_descriptor);
	if (!texture)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create a RmlUi WebGPU texture.");
		return nullptr;
	}

	UploadTexture(queue, texture, width, height, source.data(), source.size());
	return CreateTextureObject(texture, dimensions);
}

void RenderInterface_WGPU::CreateWhiteTexture()
{
	const Rml::byte white_pixel[4] = {255, 255, 255, 255};
	white_texture = CreateTexture(Rml::Span<const Rml::byte>(white_pixel, 4), Rml::Vector2i(1, 1));
	if (!white_texture)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create the RmlUi WebGPU white texture.");
		return;
	}

	// The white texture is its own fallback texture, so replace the bind group created with a temporary fallback.
	wgpuBindGroupRelease(white_texture->bind_group);
	white_texture->bind_group = CreateBindGroup(white_texture->view, white_texture->view, sampler);
}

RenderInterface_WGPU::RenderTarget RenderInterface_WGPU::CreateRenderTarget(WGPUTextureUsage extra_usage)
{
	RenderTarget target{};
	target.width = viewport_width;
	target.height = viewport_height;
	if (viewport_width == 0 || viewport_height == 0)
		return target;

	WGPUTextureDescriptor descriptor{};
	descriptor.label = MakeStringView("rmlui_wgpu_render_target");
	descriptor.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding | extra_usage;
	descriptor.dimension = WGPUTextureDimension_2D;
	descriptor.size = {viewport_width, viewport_height, 1};
	descriptor.format = target_format;
	descriptor.mipLevelCount = 1;
	descriptor.sampleCount = 1;

	target.texture = wgpuDeviceCreateTexture(device, &descriptor);
	if (!target.texture)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create a RmlUi WebGPU render target.");
		return target;
	}

	target.view = wgpuTextureCreateView(target.texture, nullptr);
	if (!target.view)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create a RmlUi WebGPU render target view.");
		wgpuTextureRelease(target.texture);
		target.texture = nullptr;
		return target;
	}

	target.bind_group = CreateBindGroup(target.view, white_texture ? white_texture->view : target.view, render_target_sampler);
	if (!target.bind_group)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create a RmlUi WebGPU render target bind group.");
		wgpuTextureViewRelease(target.view);
		wgpuTextureRelease(target.texture);
		target = {};
	}

	return target;
}

void RenderInterface_WGPU::ReleaseRenderTarget(RenderTarget& target)
{
	if (target.bind_group)
		wgpuBindGroupRelease(target.bind_group);
	if (target.view)
		wgpuTextureViewRelease(target.view);
	if (target.texture)
		wgpuTextureRelease(target.texture);
	target = {};
}

void RenderInterface_WGPU::EnsureTargetStorage()
{
	if (stencil_texture && stencil_view && root_target.view && postprocess_primary.view && postprocess_secondary.view &&
		postprocess_tertiary.view && blend_mask.view)
		return;

	if (stencil_view)
		wgpuTextureViewRelease(stencil_view);
	if (stencil_texture)
		wgpuTextureRelease(stencil_texture);
	stencil_view = nullptr;
	stencil_texture = nullptr;

	ReleaseRenderTarget(root_target);
	ReleaseRenderTarget(postprocess_primary);
	ReleaseRenderTarget(postprocess_secondary);
	ReleaseRenderTarget(postprocess_tertiary);
	ReleaseRenderTarget(blend_mask);
	for (RenderTarget& layer : layers)
		ReleaseRenderTarget(layer);
	layers.clear();

	if (viewport_width == 0 || viewport_height == 0)
		return;

	WGPUTextureDescriptor stencil_descriptor{};
	stencil_descriptor.label = MakeStringView("rmlui_wgpu_stencil");
	stencil_descriptor.usage = WGPUTextureUsage_RenderAttachment;
	stencil_descriptor.dimension = WGPUTextureDimension_2D;
	stencil_descriptor.size = {viewport_width, viewport_height, 1};
	stencil_descriptor.format = WGPUTextureFormat_Depth24PlusStencil8;
	stencil_descriptor.mipLevelCount = 1;
	stencil_descriptor.sampleCount = 1;

	stencil_texture = wgpuDeviceCreateTexture(device, &stencil_descriptor);
	stencil_view = stencil_texture ? wgpuTextureCreateView(stencil_texture, nullptr) : nullptr;
	if (!stencil_view)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create the RmlUi WebGPU stencil buffer.");
		if (stencil_texture)
			wgpuTextureRelease(stencil_texture);
		stencil_texture = nullptr;
		return;
	}

	postprocess_primary = CreateRenderTarget(WGPUTextureUsage_CopySrc);
	postprocess_secondary = CreateRenderTarget(WGPUTextureUsage_CopySrc);
	postprocess_tertiary = CreateRenderTarget(WGPUTextureUsage_CopySrc);
	blend_mask = CreateRenderTarget(WGPUTextureUsage_CopyDst);
	root_target = CreateRenderTarget(WGPUTextureUsage_CopySrc);
}

void RenderInterface_WGPU::BeginPass(RenderTarget* target, bool clear_color, WGPUColor clear_value, bool clear_stencil, uint32_t stencil_value)
{
	EndPass();

	if (!command_encoder || !target || !target->view || !stencil_view)
		return;

	current_target = target;

	WGPURenderPassColorAttachment color_attachment{};
	color_attachment.view = target->view;
	color_attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
	color_attachment.loadOp = clear_color ? WGPULoadOp_Clear : WGPULoadOp_Load;
	color_attachment.storeOp = WGPUStoreOp_Store;
	color_attachment.clearValue = clear_value;

	WGPURenderPassDepthStencilAttachment depth_stencil_attachment{};
	depth_stencil_attachment.view = stencil_view;
	depth_stencil_attachment.depthLoadOp = WGPULoadOp_Clear;
	depth_stencil_attachment.depthStoreOp = WGPUStoreOp_Store;
	depth_stencil_attachment.depthClearValue = 1.f;
	depth_stencil_attachment.depthReadOnly = false;
	depth_stencil_attachment.stencilLoadOp = clear_stencil ? WGPULoadOp_Clear : WGPULoadOp_Load;
	depth_stencil_attachment.stencilStoreOp = WGPUStoreOp_Store;
	depth_stencil_attachment.stencilClearValue = stencil_value;
	depth_stencil_attachment.stencilReadOnly = false;

	WGPURenderPassDescriptor descriptor{};
	descriptor.label = MakeStringView("rmlui_wgpu_render_pass");
	descriptor.colorAttachmentCount = 1;
	descriptor.colorAttachments = &color_attachment;
	descriptor.depthStencilAttachment = &depth_stencil_attachment;

	render_pass = wgpuCommandEncoderBeginRenderPass(command_encoder, &descriptor);
	if (!render_pass)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to begin an RmlUi WebGPU render pass.");
	}
}

void RenderInterface_WGPU::EndPass()
{
	if (render_pass)
	{
		wgpuRenderPassEncoderEnd(render_pass);
		wgpuRenderPassEncoderRelease(render_pass);
		render_pass = nullptr;
	}
}

RenderInterface_WGPU::RenderTarget* RenderInterface_WGPU::GetTarget(Rml::LayerHandle handle)
{
	if (handle == 0)
		return &root_target;

	const size_t index = size_t(handle) - 1;
	if (index < layers.size() && index < layer_depth)
		return &layers[index];
	return nullptr;
}

void RenderInterface_WGPU::SetScissor()
{
	if (!render_pass)
		return;

	int32_t x = 0, y = 0;
	int32_t x1 = int32_t(viewport_width), y1 = int32_t(viewport_height);

	if (scissor_enabled)
	{
		x = std::max(scissor_region.Left(), 0);
		y = std::max(scissor_region.Top(), 0);
		x1 = std::min(scissor_region.Right(), int32_t(viewport_width));
		y1 = std::min(scissor_region.Bottom(), int32_t(viewport_height));
		if (x1 <= x || y1 <= y)
		{
			x = 0;
			y = 0;
			x1 = 0;
			y1 = 0;
		}
	}

	wgpuRenderPassEncoderSetScissorRect(render_pass, uint32_t(x), uint32_t(y), uint32_t(x1 - x), uint32_t(y1 - y));
}

bool RenderInterface_WGPU::WriteUniform(const DrawUniforms& uniforms, uint32_t& out_offset)
{
	if (uniform_offset + UNIFORM_STRIDE > UNIFORM_BUFFER_SIZE)
	{
		if (!uniform_overflow_reported)
		{
			Rml::Log::Message(Rml::Log::LT_ERROR, "RmlUi WebGPU uniform buffer overflow: too many draw calls in a single frame.");
			uniform_overflow_reported = true;
		}
		return false;
	}

	wgpuQueueWriteBuffer(queue, uniform_buffer, uniform_offset, &uniforms, sizeof(DrawUniforms));
	out_offset = static_cast<uint32_t>(uniform_offset);
	uniform_offset += UNIFORM_STRIDE;
	return true;
}

void RenderInterface_WGPU::DrawGeometryWithPipeline(WGPURenderPipeline pipeline, Geometry* geometry, Rml::Vector2f translation,
	WGPUBindGroup bind_group, bool clipped)
{
	if (!render_pass || !geometry || !bind_group || !pipeline)
		return;

	DrawUniforms uniforms{};
	uniforms.transform = transform;
	uniforms.translation = translation;

	uint32_t offset = 0;
	if (!WriteUniform(uniforms, offset))
		return;

	SetScissor();
	wgpuRenderPassEncoderSetPipeline(render_pass, pipeline);
	if (clipped)
		wgpuRenderPassEncoderSetStencilReference(render_pass, clip_mask_ref);
	wgpuRenderPassEncoderSetBindGroup(render_pass, 0, bind_group, 1, &offset);
	wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, geometry->vertex_buffer, 0, geometry->vertex_buffer_size);
	wgpuRenderPassEncoderSetIndexBuffer(render_pass, geometry->index_buffer, WGPUIndexFormat_Uint32, 0, geometry->index_buffer_size);
	wgpuRenderPassEncoderDrawIndexed(render_pass, geometry->index_count, 1, 0, 0, 0);
}

void RenderInterface_WGPU::DrawFullscreenWithPipeline(WGPURenderPipeline pipeline, WGPUBindGroup bind_group, const DrawUniforms& uniforms)
{
	if (!render_pass || !bind_group || !pipeline || !fullscreen_quad)
		return;

	Geometry* geometry = reinterpret_cast<Geometry*>(fullscreen_quad);
	uint32_t offset = 0;
	if (!WriteUniform(uniforms, offset))
		return;

	SetScissor();
	wgpuRenderPassEncoderSetPipeline(render_pass, pipeline);
	wgpuRenderPassEncoderSetBindGroup(render_pass, 0, bind_group, 1, &offset);
	wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, geometry->vertex_buffer, 0, geometry->vertex_buffer_size);
	wgpuRenderPassEncoderSetIndexBuffer(render_pass, geometry->index_buffer, WGPUIndexFormat_Uint32, 0, geometry->index_buffer_size);
	wgpuRenderPassEncoderDrawIndexed(render_pass, geometry->index_count, 1, 0, 0, 0);
}

WGPURenderPipeline RenderInterface_WGPU::ChooseGeometryPipeline(WGPURenderPipeline normal, WGPURenderPipeline clipped) const
{
	return clip_mask_enabled ? clipped : normal;
}

void RenderInterface_WGPU::BeginFrame(WGPUCommandEncoder encoder, WGPUTexture surface_texture, WGPUTextureView surface_view, uint32_t width, uint32_t height)
{
	EndPass();
	ReleasePendingResources();

	command_encoder = encoder;
	this->surface_texture = surface_texture;
	viewport_width = width;
	viewport_height = height;

	if (width != allocated_width || height != allocated_height)
	{
		ReleaseRenderTarget(postprocess_primary);
		ReleaseRenderTarget(postprocess_secondary);
		ReleaseRenderTarget(postprocess_tertiary);
		ReleaseRenderTarget(blend_mask);
		ReleaseRenderTarget(root_target);
		for (RenderTarget& layer : layers)
			ReleaseRenderTarget(layer);
		layers.clear();

		if (stencil_view)
			wgpuTextureViewRelease(stencil_view);
		if (stencil_texture)
			wgpuTextureRelease(stencil_texture);
		stencil_view = nullptr;
		stencil_texture = nullptr;

		allocated_width = width;
		allocated_height = height;
	}

	surface_target = {};
	surface_target.view = surface_view;
	surface_target.width = width;
	surface_target.height = height;
	current_target = &root_target;
	layer_depth = 0;

	if (width == 0 || height == 0 || !surface_view || !command_encoder)
		return;

	EnsureTargetStorage();
	if (!root_target.view)
		return;
	root_target.width = width;
	root_target.height = height;

	uniform_offset = 0;
	uniform_overflow_reported = false;
	projection = Rml::Matrix4f::ProjectOrtho(0.f, float(width), float(height), 0.f, -10000.f, 10000.f);
	transform = projection;
	scissor_region = Rml::Rectanglei::FromCorners({0, 0}, {int(width), int(height)});
	scissor_enabled = false;
	clip_mask_enabled = false;
	clip_mask_ref = 1;

	BeginPass(&root_target, true, transparent_clear, true, 0);
}

void RenderInterface_WGPU::SetBackgroundCallback(BackgroundCallback callback)
{
	background_callback = std::move(callback);
}

void RenderInterface_WGPU::EndFrame()
{
	EndPass();

	if (command_encoder && surface_target.view && root_target.view && viewport_width > 0 && viewport_height > 0)
	{
		if (background_callback)
			background_callback(command_encoder, surface_texture, surface_target.view, viewport_width, viewport_height);

		// The UI is rendered into a transparent offscreen root layer so that a background callback can draw behind it, and so that
		// layers can sample it (e.g. for backdrop filters). Composite it over the background at the end of the frame.
		scissor_enabled = false;
		BeginPass(&surface_target, !background_callback, clear_color, false, 0);
		DrawFullscreenWithPipeline(pipeline_passthrough_blend, root_target.bind_group, DrawUniforms{});
		EndPass();
	}
}

Rml::CompiledGeometryHandle RenderInterface_WGPU::CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices)
{
	if (vertices.empty() || indices.empty() || !device || !queue)
		return 0;

	Geometry* geometry = new Geometry();
	geometry->vertex_buffer_size = vertices.size() * sizeof(Rml::Vertex);
	geometry->index_buffer_size = indices.size() * sizeof(int);
	geometry->index_count = static_cast<uint32_t>(indices.size());

	WGPUBufferDescriptor vertex_buffer_descriptor{};
	vertex_buffer_descriptor.label = MakeStringView("rmlui_wgpu_vertex_buffer");
	vertex_buffer_descriptor.size = geometry->vertex_buffer_size;
	vertex_buffer_descriptor.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
	geometry->vertex_buffer = wgpuDeviceCreateBuffer(device, &vertex_buffer_descriptor);

	WGPUBufferDescriptor index_buffer_descriptor{};
	index_buffer_descriptor.label = MakeStringView("rmlui_wgpu_index_buffer");
	index_buffer_descriptor.size = geometry->index_buffer_size;
	index_buffer_descriptor.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
	geometry->index_buffer = wgpuDeviceCreateBuffer(device, &index_buffer_descriptor);

	if (!geometry->vertex_buffer || !geometry->index_buffer)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create RmlUi WebGPU geometry buffers.");
		if (geometry->vertex_buffer)
			wgpuBufferRelease(geometry->vertex_buffer);
		if (geometry->index_buffer)
			wgpuBufferRelease(geometry->index_buffer);
		delete geometry;
		return 0;
	}

	wgpuQueueWriteBuffer(queue, geometry->vertex_buffer, 0, vertices.data(), geometry->vertex_buffer_size);
	wgpuQueueWriteBuffer(queue, geometry->index_buffer, 0, indices.data(), geometry->index_buffer_size);

	return reinterpret_cast<Rml::CompiledGeometryHandle>(geometry);
}

void RenderInterface_WGPU::ReleaseGeometry(Rml::CompiledGeometryHandle handle)
{
	Geometry* geometry = reinterpret_cast<Geometry*>(handle);
	if (!geometry)
		return;

	pending_release_buffers.push_back(geometry->vertex_buffer);
	pending_release_buffers.push_back(geometry->index_buffer);
	delete geometry;
}

void RenderInterface_WGPU::RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation, Rml::TextureHandle texture_handle)
{
	Geometry* geometry = reinterpret_cast<Geometry*>(handle);
	Texture* texture = texture_handle ? reinterpret_cast<Texture*>(texture_handle) : white_texture;
	if (!geometry || !texture)
		return;

	DrawGeometryWithPipeline(ChooseGeometryPipeline(pipeline_main, pipeline_main_clipped), geometry, translation, texture->bind_group,
		clip_mask_enabled);
}

void RenderInterface_WGPU::EnableScissorRegion(bool enable)
{
	scissor_enabled = enable;
}

void RenderInterface_WGPU::SetScissorRegion(Rml::Rectanglei region)
{
	scissor_region = region;
}

void RenderInterface_WGPU::SetTransform(const Rml::Matrix4f* new_transform)
{
	transform = new_transform ? (projection * (*new_transform)) : projection;
}

void RenderInterface_WGPU::EnableClipMask(bool enable)
{
	clip_mask_enabled = enable;
}

void RenderInterface_WGPU::RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle handle, Rml::Vector2f translation)
{
	Geometry* geometry = reinterpret_cast<Geometry*>(handle);
	if (!render_pass || !geometry || !white_texture)
		return;

	WGPURenderPipeline mask_pipeline = pipeline_mask_replace;
	uint32_t write_reference = 1;

	switch (operation)
	{
	case Rml::ClipMaskOperation::Set:
		clip_mask_ref = 1;
		write_reference = 1;
		mask_pipeline = pipeline_mask_replace;
		BeginPass(current_target, false, {}, true, 0);
		break;
	case Rml::ClipMaskOperation::SetInverse:
		clip_mask_ref = 1;
		write_reference = 0;
		mask_pipeline = pipeline_mask_replace;
		BeginPass(current_target, false, {}, true, 1);
		break;
	case Rml::ClipMaskOperation::Intersect:
		clip_mask_ref = std::min(clip_mask_ref + 1, 255u);
		write_reference = clip_mask_ref;
		mask_pipeline = pipeline_mask_increment;
		break;
	}

	if (!render_pass || !mask_pipeline)
		return;

	DrawUniforms uniforms{};
	uniforms.transform = transform;
	uniforms.translation = translation;

	uint32_t offset = 0;
	if (!WriteUniform(uniforms, offset))
		return;

	SetScissor();
	wgpuRenderPassEncoderSetPipeline(render_pass, mask_pipeline);
	wgpuRenderPassEncoderSetStencilReference(render_pass, write_reference);
	wgpuRenderPassEncoderSetBindGroup(render_pass, 0, white_texture->bind_group, 1, &offset);
	wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, geometry->vertex_buffer, 0, geometry->vertex_buffer_size);
	wgpuRenderPassEncoderSetIndexBuffer(render_pass, geometry->index_buffer, WGPUIndexFormat_Uint32, 0, geometry->index_buffer_size);
	wgpuRenderPassEncoderDrawIndexed(render_pass, geometry->index_count, 1, 0, 0, 0);
}

Rml::TextureHandle RenderInterface_WGPU::LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source)
{
	Rml::String file_data;
	if (!Rml::GetFileInterface()->LoadFile(source, file_data))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to load texture file: %s", source.c_str());
		return 0;
	}

	const size_t extension_index = source.rfind('.');
	Rml::String extension = (extension_index == Rml::String::npos) ? Rml::String() : source.substr(extension_index + 1);
	for (char& c : extension)
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

	if (extension != "tga")
	{
		Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported texture format '%s' for file: %s", extension.c_str(), source.c_str());
		return 0;
	}

	std::vector<Rml::byte> pixels;
	if (!DecodeTGA(reinterpret_cast<const Rml::byte*>(file_data.data()), file_data.size(), texture_dimensions, pixels))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to decode TGA texture file: %s", source.c_str());
		return 0;
	}

	Texture* texture = CreateTexture(Rml::Span<const Rml::byte>(pixels.data(), pixels.size()), texture_dimensions);
	if (!texture)
		return 0;

	return reinterpret_cast<Rml::TextureHandle>(texture);
}

Rml::TextureHandle RenderInterface_WGPU::GenerateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i source_dimensions)
{
	Texture* texture = CreateTexture(source, source_dimensions);
	if (!texture)
		return 0;

	return reinterpret_cast<Rml::TextureHandle>(texture);
}

void RenderInterface_WGPU::ReleaseTexture(Rml::TextureHandle handle)
{
	Texture* texture = reinterpret_cast<Texture*>(handle);
	if (!texture)
		return;

	pending_release_bind_groups.push_back(texture->bind_group);
	pending_release_texture_views.push_back(texture->view);
	pending_release_textures.push_back(texture->texture);
	delete texture;
}

Rml::LayerHandle RenderInterface_WGPU::PushLayer()
{
	if (!render_pass || !current_target)
		return 0;

	layer_depth++;
	if (layers.size() < layer_depth)
		layers.push_back(CreateRenderTarget(WGPUTextureUsage_CopySrc));

	RenderTarget* target = &layers[layer_depth - 1];
	if (!target->view)
	{
		layer_depth--;
		return 0;
	}

	BeginPass(target, true, WGPUColor{0.0, 0.0, 0.0, 0.0}, false, 0);
	return static_cast<Rml::LayerHandle>(layer_depth);
}

void RenderInterface_WGPU::PopLayer()
{
	if (layer_depth == 0)
		return;

	layer_depth--;
	current_target = (layer_depth == 0) ? &root_target : &layers[layer_depth - 1];
	BeginPass(current_target, false, WGPUColor{}, false, 0);
}

void RenderInterface_WGPU::CompositeLayers(Rml::LayerHandle source_handle, Rml::LayerHandle destination_handle, Rml::BlendMode blend_mode,
	Rml::Span<const Rml::CompiledFilterHandle> filters)
{
	RenderTarget* top = current_target;
	RenderTarget* source = GetTarget(source_handle);
	RenderTarget* destination = GetTarget(destination_handle);
	if (!top || !source || !destination || !postprocess_primary.view)
		return;

	if (!source->bind_group)
	{
		Rml::Log::Message(Rml::Log::LT_WARNING, "The SDL_WGPU backend cannot composite from a render target without a texture binding.");
		return;
	}

	EndPass();
	BeginPass(&postprocess_primary, true, WGPUColor{0.0, 0.0, 0.0, 0.0}, false, 0);
	DrawFullscreenWithPipeline(pipeline_passthrough_replace, source->bind_group, DrawUniforms{});

	ApplyFilters(filters);

	BeginPass(destination, false, WGPUColor{}, false, 0);
	DrawFullscreenWithPipeline(blend_mode == Rml::BlendMode::Blend ? pipeline_passthrough_blend : pipeline_passthrough_replace,
		postprocess_primary.bind_group, DrawUniforms{});

	if (top != destination)
		BeginPass(top, false, WGPUColor{}, false, 0);

	current_target = top;
}

Rml::TextureHandle RenderInterface_WGPU::SaveLayerAsTexture()
{
	if (!current_target || !current_target->texture || !command_encoder)
	{
		Rml::Log::Message(Rml::Log::LT_WARNING, "SaveLayerAsTexture requires an active offscreen render layer.");
		return 0;
	}

	Rml::Rectanglei bounds = scissor_enabled ? scissor_region : Rml::Rectanglei::FromCorners({0, 0}, {int(viewport_width), int(viewport_height)});
	bounds.p0.x = Rml::Math::Clamp(bounds.p0.x, 0, int(viewport_width));
	bounds.p0.y = Rml::Math::Clamp(bounds.p0.y, 0, int(viewport_height));
	bounds.p1.x = Rml::Math::Clamp(bounds.p1.x, 0, int(viewport_width));
	bounds.p1.y = Rml::Math::Clamp(bounds.p1.y, 0, int(viewport_height));
	if (!bounds.Valid() || bounds.Width() == 0 || bounds.Height() == 0)
		return 0;

	WGPUTextureDescriptor descriptor{};
	descriptor.label = MakeStringView("rmlui_wgpu_saved_layer");
	descriptor.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
	descriptor.dimension = WGPUTextureDimension_2D;
	descriptor.size = {uint32_t(bounds.Width()), uint32_t(bounds.Height()), 1};
	descriptor.format = target_format;
	descriptor.mipLevelCount = 1;
	descriptor.sampleCount = 1;

	WGPUTexture saved_texture = wgpuDeviceCreateTexture(device, &descriptor);
	if (!saved_texture)
		return 0;

	Texture* saved_layer = CreateTextureObject(saved_texture, bounds.Size());
	if (!saved_layer)
		return 0;

	EndPass();

	WGPUTexelCopyTextureInfo source{};
	source.texture = current_target->texture;
	source.mipLevel = 0;
	source.origin = {uint32_t(bounds.Left()), uint32_t(bounds.Top()), 0};
	source.aspect = WGPUTextureAspect_All;

	WGPUTexelCopyTextureInfo destination{};
	destination.texture = saved_texture;
	destination.mipLevel = 0;
	destination.origin = {0, 0, 0};
	destination.aspect = WGPUTextureAspect_All;

	WGPUExtent3D extent = {uint32_t(bounds.Width()), uint32_t(bounds.Height()), 1};
	wgpuCommandEncoderCopyTextureToTexture(command_encoder, &source, &destination, &extent);

	BeginPass(current_target, false, WGPUColor{}, false, 0);
	return reinterpret_cast<Rml::TextureHandle>(saved_layer);
}

Rml::CompiledFilterHandle RenderInterface_WGPU::SaveLayerAsMaskImage()
{
	if (!current_target || !current_target->texture || !blend_mask.texture || !command_encoder)
	{
		Rml::Log::Message(Rml::Log::LT_WARNING, "SaveLayerAsMaskImage requires an active offscreen render layer.");
		return 0;
	}

	Rml::Rectanglei bounds = scissor_enabled ? scissor_region : Rml::Rectanglei::FromCorners({0, 0}, {int(viewport_width), int(viewport_height)});
	bounds.p0.x = Rml::Math::Clamp(bounds.p0.x, 0, int(viewport_width));
	bounds.p0.y = Rml::Math::Clamp(bounds.p0.y, 0, int(viewport_height));
	bounds.p1.x = Rml::Math::Clamp(bounds.p1.x, 0, int(viewport_width));
	bounds.p1.y = Rml::Math::Clamp(bounds.p1.y, 0, int(viewport_height));
	if (!bounds.Valid() || bounds.Width() == 0 || bounds.Height() == 0)
		return 0;

	EndPass();

	WGPUTexelCopyTextureInfo source{};
	source.texture = current_target->texture;
	source.origin = {uint32_t(bounds.Left()), uint32_t(bounds.Top()), 0};
	source.aspect = WGPUTextureAspect_All;

	WGPUTexelCopyTextureInfo destination{};
	destination.texture = blend_mask.texture;
	destination.origin = {uint32_t(bounds.Left()), uint32_t(bounds.Top()), 0};
	destination.aspect = WGPUTextureAspect_All;

	WGPUExtent3D extent = {uint32_t(bounds.Width()), uint32_t(bounds.Height()), 1};
	wgpuCommandEncoderCopyTextureToTexture(command_encoder, &source, &destination, &extent);

	BeginPass(current_target, false, WGPUColor{}, false, 0);

	CompiledFilter* filter = new CompiledFilter();
	filter->type = FilterType::MaskImage;
	return reinterpret_cast<Rml::CompiledFilterHandle>(filter);
}

Rml::CompiledFilterHandle RenderInterface_WGPU::CompileFilter(const Rml::String& name, const Rml::Dictionary& parameters)
{
	CompiledFilter filter;

	if (name == "opacity")
	{
		filter.type = FilterType::Passthrough;
		filter.blend_factor = Rml::Get(parameters, "value", 1.0f);
	}
	else if (name == "blur")
	{
		filter.type = FilterType::Blur;
		filter.sigma = Rml::Get(parameters, "sigma", 1.0f);
	}
	else if (name == "drop-shadow")
	{
		filter.type = FilterType::DropShadow;
		filter.sigma = Rml::Get(parameters, "sigma", 0.f);
		const Rml::ColourbPremultiplied color = Rml::Get(parameters, "color", Rml::Colourb()).ToPremultiplied();
		filter.color = {color[0] / 255.f, color[1] / 255.f, color[2] / 255.f, color[3] / 255.f};
		filter.offset = Rml::Get(parameters, "offset", Rml::Vector2f(0.f));
	}
	else if (name == "brightness")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		filter.color_matrix = Rml::Matrix4f::Diag(value, value, value, 1.f);
	}
	else if (name == "contrast")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float grayness = 0.5f - 0.5f * value;
		filter.color_matrix = Rml::Matrix4f::Diag(value, value, value, 1.f);
		filter.color_matrix.SetColumn(3, Rml::Vector4f(grayness, grayness, grayness, 1.f));
	}
	else if (name == "invert")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Math::Clamp(Rml::Get(parameters, "value", 1.0f), 0.f, 1.f);
		const float inverted = 1.f - 2.f * value;
		filter.color_matrix = Rml::Matrix4f::Diag(inverted, inverted, inverted, 1.f);
		filter.color_matrix.SetColumn(3, Rml::Vector4f(value, value, value, 1.f));
	}
	else if (name == "grayscale")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float rev_value = 1.f - value;
		const Rml::Vector3f gray = value * Rml::Vector3f(0.2126f, 0.7152f, 0.0722f);
		filter.color_matrix = Rml::Matrix4f::FromRows({gray.x + rev_value, gray.y, gray.z, 0.f}, {gray.x, gray.y + rev_value, gray.z, 0.f},
			{gray.x, gray.y, gray.z + rev_value, 0.f}, {0.f, 0.f, 0.f, 1.f});
	}
	else if (name == "sepia")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float rev_value = 1.f - value;
		const Rml::Vector3f r_mix = value * Rml::Vector3f(0.393f, 0.769f, 0.189f);
		const Rml::Vector3f g_mix = value * Rml::Vector3f(0.349f, 0.686f, 0.168f);
		const Rml::Vector3f b_mix = value * Rml::Vector3f(0.272f, 0.534f, 0.131f);
		filter.color_matrix = Rml::Matrix4f::FromRows({r_mix.x + rev_value, r_mix.y, r_mix.z, 0.f}, {g_mix.x, g_mix.y + rev_value, g_mix.z, 0.f},
			{b_mix.x, b_mix.y, b_mix.z + rev_value, 0.f}, {0.f, 0.f, 0.f, 1.f});
	}
	else if (name == "hue-rotate")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float s = Rml::Math::Sin(value);
		const float c = Rml::Math::Cos(value);
		filter.color_matrix = Rml::Matrix4f::FromRows(
			{0.213f + 0.787f * c - 0.213f * s, 0.715f - 0.715f * c - 0.715f * s, 0.072f - 0.072f * c + 0.928f * s, 0.f},
			{0.213f - 0.213f * c + 0.143f * s, 0.715f + 0.285f * c + 0.140f * s, 0.072f - 0.072f * c - 0.283f * s, 0.f},
			{0.213f - 0.213f * c - 0.787f * s, 0.715f - 0.715f * c + 0.715f * s, 0.072f + 0.928f * c + 0.072f * s, 0.f},
			{0.f, 0.f, 0.f, 1.f});
	}
	else if (name == "saturate")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		filter.color_matrix = Rml::Matrix4f::FromRows(
			{0.213f + 0.787f * value, 0.715f - 0.715f * value, 0.072f - 0.072f * value, 0.f},
			{0.213f - 0.213f * value, 0.715f + 0.285f * value, 0.072f - 0.072f * value, 0.f},
			{0.213f - 0.213f * value, 0.715f - 0.715f * value, 0.072f + 0.928f * value, 0.f},
			{0.f, 0.f, 0.f, 1.f});
	}

	if (filter.type == FilterType::Invalid)
	{
		Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported filter type '%s'.", name.c_str());
		return 0;
	}

	return reinterpret_cast<Rml::CompiledFilterHandle>(new CompiledFilter(std::move(filter)));
}

void RenderInterface_WGPU::ReleaseFilter(Rml::CompiledFilterHandle handle)
{
	delete reinterpret_cast<CompiledFilter*>(handle);
}

void RenderInterface_WGPU::ApplyFilters(Rml::Span<const Rml::CompiledFilterHandle> filters)
{
	for (const Rml::CompiledFilterHandle handle : filters)
	{
		const CompiledFilter* filter = reinterpret_cast<const CompiledFilter*>(handle);
		if (filter && filter->type != FilterType::Invalid)
			ApplyFilter(*filter);
	}
}

void RenderInterface_WGPU::ApplyFilter(const CompiledFilter& filter)
{
	if (!postprocess_primary.view)
		return;

	const int x0 = scissor_enabled ? Rml::Math::Clamp(scissor_region.Left(), 0, int(viewport_width)) : 0;
	const int y0 = scissor_enabled ? Rml::Math::Clamp(scissor_region.Top(), 0, int(viewport_height)) : 0;
	const int x1 = scissor_enabled ? Rml::Math::Clamp(scissor_region.Right(), 0, int(viewport_width)) : int(viewport_width);
	const int y1 = scissor_enabled ? Rml::Math::Clamp(scissor_region.Bottom(), 0, int(viewport_height)) : int(viewport_height);
	if (x1 <= x0 || y1 <= y0)
		return;

	auto set_bounds = [&](DrawUniforms& uniforms) {
		// Offset by half-texel values so that texture lookups are clamped to fragment centers, thereby avoiding color bleeding
		// from neighboring texels due to bilinear interpolation.
		uniforms.tex_coord_min = {(float(x0) + 0.5f) / float(viewport_width), (float(y0) + 0.5f) / float(viewport_height)};
		uniforms.tex_coord_max = {(float(x1) - 0.5f) / float(viewport_width), (float(y1) - 0.5f) / float(viewport_height)};
	};

	auto set_blur_weights = [](DrawUniforms& uniforms, float sigma) {
		sigma = Rml::Math::Absolute(sigma);
		float weights[4] = {};
		float normalization = 0.f;
		for (int i = 0; i < 4; i++)
		{
			weights[i] = (sigma < 0.1f) ? float(i == 0) : Rml::Math::Exp(-float(i * i) / (2.f * sigma * sigma)) / (Rml::Math::SquareRoot(2.f * Rml::Math::RMLUI_PI) * sigma);
			normalization += (i == 0 ? 1.f : 2.f) * weights[i];
		}
		uniforms.blur_weights = {weights[0] / normalization, weights[1] / normalization, weights[2] / normalization, weights[3] / normalization};
	};

	auto draw = [&](WGPURenderPipeline pipeline, RenderTarget& destination, WGPUBindGroup source, const DrawUniforms& uniforms) {
		BeginPass(&destination, false, WGPUColor{}, false, 0);
		DrawFullscreenWithPipeline(pipeline, source, uniforms);
	};

	switch (filter.type)
	{
	case FilterType::Passthrough:
	{
		DrawUniforms uniforms{};
		uniforms.opacity = filter.blend_factor;
		draw(pipeline_opacity, postprocess_secondary, postprocess_primary.bind_group, uniforms);
		std::swap(postprocess_primary, postprocess_secondary);
	}
	break;
	case FilterType::Blur:
	{
		if (Rml::Math::Absolute(filter.sigma) < 0.05f)
			break;

		DrawUniforms uniforms{};
		set_bounds(uniforms);
		set_blur_weights(uniforms, filter.sigma);

		uniforms.texel_offset = {0.f, 1.f / float(viewport_height)};
		draw(pipeline_blur, postprocess_secondary, postprocess_primary.bind_group, uniforms);

		uniforms.texel_offset = {1.f / float(viewport_width), 0.f};
		draw(pipeline_blur, postprocess_primary, postprocess_secondary.bind_group, uniforms);
	}
	break;
	case FilterType::DropShadow:
	{
		DrawUniforms uniforms{};
		set_bounds(uniforms);
		uniforms.color = filter.color;
		// The shader samples at tex_coord + texel_offset, so negate the offset to move the shadow in the requested direction.
		uniforms.texel_offset = {-filter.offset.x / float(viewport_width), -filter.offset.y / float(viewport_height)};
		draw(pipeline_drop_shadow, postprocess_secondary, postprocess_primary.bind_group, uniforms);

		if (Rml::Math::Absolute(filter.sigma) >= 0.5f)
		{
			DrawUniforms blur_uniforms{};
			set_bounds(blur_uniforms);
			set_blur_weights(blur_uniforms, filter.sigma);

			blur_uniforms.texel_offset = {0.f, 1.f / float(viewport_height)};
			draw(pipeline_blur, postprocess_tertiary, postprocess_secondary.bind_group, blur_uniforms);

			blur_uniforms.texel_offset = {1.f / float(viewport_width), 0.f};
			draw(pipeline_blur, postprocess_secondary, postprocess_tertiary.bind_group, blur_uniforms);
		}

		draw(pipeline_passthrough_blend, postprocess_secondary, postprocess_primary.bind_group, DrawUniforms{});
		std::swap(postprocess_primary, postprocess_secondary);
	}
	break;
	case FilterType::ColorMatrix:
	{
		DrawUniforms uniforms{};
		uniforms.color_matrix = filter.color_matrix;
		draw(pipeline_color_matrix, postprocess_secondary, postprocess_primary.bind_group, uniforms);
		std::swap(postprocess_primary, postprocess_secondary);
	}
	break;
	case FilterType::MaskImage:
	{
		WGPUBindGroup mask_bind_group = CreateBindGroup(postprocess_primary.view, blend_mask.view, render_target_sampler);
		if (!mask_bind_group)
			break;
		draw(pipeline_blend_mask, postprocess_secondary, mask_bind_group, DrawUniforms{});
		std::swap(postprocess_primary, postprocess_secondary);
		pending_release_bind_groups.push_back(mask_bind_group);
	}
	break;
	case FilterType::Invalid:
		break;
	}
}

Rml::CompiledShaderHandle RenderInterface_WGPU::CompileShader(const Rml::String& name, const Rml::Dictionary& parameters)
{
	CompiledShader shader;

	auto apply_stop_list = [](CompiledShader& target, const Rml::Dictionary& shader_parameters) {
		auto it = shader_parameters.find("color_stop_list");
		if (it == shader_parameters.end() || it->second.GetType() != Rml::Variant::COLORSTOPLIST)
			return;

		const Rml::ColorStopList& color_stop_list = it->second.GetReference<Rml::ColorStopList>();
		const uint32_t num_stops = uint32_t(Rml::Math::Min(int(color_stop_list.size()), int(MAX_NUM_STOPS)));
		target.num_stops = num_stops;
		for (uint32_t i = 0; i < num_stops; i++)
		{
			const Rml::ColorStop& stop = color_stop_list[i];
			target.stop_positions[i] = stop.position.number;
			const Rml::ColourbPremultiplied& color = stop.color;
			target.stop_colors[i] = {color[0] / 255.f, color[1] / 255.f, color[2] / 255.f, color[3] / 255.f};
		}
	};

	if (name == "linear-gradient")
	{
		const bool repeating = Rml::Get(parameters, "repeating", false);
		shader.type = repeating ? ShaderType::RepeatingLinearGradient : ShaderType::LinearGradient;
		shader.func = repeating ? 3u : 0u;
		shader.p = Rml::Get(parameters, "p0", Rml::Vector2f(0.f));
		shader.v = Rml::Get(parameters, "p1", Rml::Vector2f(0.f)) - shader.p;
		apply_stop_list(shader, parameters);
	}
	else if (name == "radial-gradient")
	{
		const bool repeating = Rml::Get(parameters, "repeating", false);
		shader.type = repeating ? ShaderType::RepeatingRadialGradient : ShaderType::RadialGradient;
		shader.func = repeating ? 4u : 1u;
		shader.p = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
		shader.v = Rml::Vector2f(1.f) / Rml::Get(parameters, "radius", Rml::Vector2f(1.f));
		apply_stop_list(shader, parameters);
	}
	else if (name == "conic-gradient")
	{
		const bool repeating = Rml::Get(parameters, "repeating", false);
		shader.type = repeating ? ShaderType::RepeatingConicGradient : ShaderType::ConicGradient;
		shader.func = repeating ? 5u : 2u;
		shader.p = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
		const float angle = Rml::Get(parameters, "angle", 0.f);
		shader.v = {Rml::Math::Cos(angle), Rml::Math::Sin(angle)};
		apply_stop_list(shader, parameters);
	}
	else if (name == "shader")
	{
		const Rml::String value = Rml::Get(parameters, "value", Rml::String());
		if (value == "creation")
		{
			shader.type = ShaderType::Creation;
			shader.dimensions = Rml::Get(parameters, "dimensions", Rml::Vector2f(0.f));
		}
	}

	if (shader.type == ShaderType::Invalid)
	{
		Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported shader type '%s'.", name.c_str());
		return 0;
	}

	return reinterpret_cast<Rml::CompiledShaderHandle>(new CompiledShader(std::move(shader)));
}

void RenderInterface_WGPU::RenderShader(Rml::CompiledShaderHandle shader_handle, Rml::CompiledGeometryHandle geometry_handle,
	Rml::Vector2f translation, Rml::TextureHandle /*texture*/)
{
	CompiledShader* shader = reinterpret_cast<CompiledShader*>(shader_handle);
	Geometry* geometry = reinterpret_cast<Geometry*>(geometry_handle);
	if (!shader || !geometry || !white_texture || !render_pass)
		return;

	DrawUniforms uniforms{};
	uniforms.transform = transform;
	uniforms.translation = translation;

	WGPURenderPipeline pipeline = nullptr;
	if (shader->type == ShaderType::LinearGradient || shader->type == ShaderType::RadialGradient || shader->type == ShaderType::ConicGradient ||
		shader->type == ShaderType::RepeatingLinearGradient || shader->type == ShaderType::RepeatingRadialGradient ||
		shader->type == ShaderType::RepeatingConicGradient)
	{
		pipeline = ChooseGeometryPipeline(pipeline_gradient, pipeline_gradient_clipped);
		uniforms.func = shader->func;
		uniforms.p = shader->p;
		uniforms.v = shader->v;
		uniforms.num_stops = shader->num_stops;
		for (uint32_t i = 0; i < shader->num_stops; i++)
		{
			uniforms.stop_positions[i] = shader->stop_positions[i];
			uniforms.stop_colors[i] = shader->stop_colors[i];
		}
	}
	else if (shader->type == ShaderType::Creation)
	{
		pipeline = ChooseGeometryPipeline(pipeline_creation, pipeline_creation_clipped);
		uniforms.dimensions = shader->dimensions;
		if (Rml::SystemInterface* system_interface = Rml::GetSystemInterface())
			uniforms.time = static_cast<float>(system_interface->GetElapsedTime());
	}

	if (!pipeline)
		return;

	uint32_t offset = 0;
	if (!WriteUniform(uniforms, offset))
		return;

	SetScissor();
	wgpuRenderPassEncoderSetPipeline(render_pass, pipeline);
	if (clip_mask_enabled)
		wgpuRenderPassEncoderSetStencilReference(render_pass, clip_mask_ref);
	wgpuRenderPassEncoderSetBindGroup(render_pass, 0, white_texture->bind_group, 1, &offset);
	wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, geometry->vertex_buffer, 0, geometry->vertex_buffer_size);
	wgpuRenderPassEncoderSetIndexBuffer(render_pass, geometry->index_buffer, WGPUIndexFormat_Uint32, 0, geometry->index_buffer_size);
	wgpuRenderPassEncoderDrawIndexed(render_pass, geometry->index_count, 1, 0, 0, 0);
}

void RenderInterface_WGPU::ReleaseShader(Rml::CompiledShaderHandle handle)
{
	delete reinterpret_cast<CompiledShader*>(handle);
}

void RenderInterface_WGPU::ReleasePendingResources()
{
	if (!device)
		return;

	for (WGPUBindGroup bind_group : pending_release_bind_groups)
	{
		if (bind_group)
			wgpuBindGroupRelease(bind_group);
	}
	pending_release_bind_groups.clear();

	for (WGPUTextureView view : pending_release_texture_views)
	{
		if (view)
			wgpuTextureViewRelease(view);
	}
	pending_release_texture_views.clear();

	for (WGPUTexture texture : pending_release_textures)
	{
		if (texture)
			wgpuTextureRelease(texture);
	}
	pending_release_textures.clear();

	for (WGPUBuffer buffer : pending_release_buffers)
	{
		if (buffer)
			wgpuBufferRelease(buffer);
	}
	pending_release_buffers.clear();
}

void RenderInterface_WGPU::Shutdown()
{
	EndPass();

	if (!device)
		return;

	ReleasePendingResources();

	ReleaseRenderTarget(root_target);
	ReleaseRenderTarget(postprocess_primary);
	ReleaseRenderTarget(postprocess_secondary);
	ReleaseRenderTarget(postprocess_tertiary);
	ReleaseRenderTarget(blend_mask);
	for (RenderTarget& layer : layers)
		ReleaseRenderTarget(layer);
	layers.clear();

	if (stencil_view)
		wgpuTextureViewRelease(stencil_view);
	if (stencil_texture)
		wgpuTextureRelease(stencil_texture);
	stencil_view = nullptr;
	stencil_texture = nullptr;

	if (fullscreen_quad)
	{
		ReleaseGeometry(fullscreen_quad);
		fullscreen_quad = 0;
	}

	if (white_texture)
	{
		wgpuBindGroupRelease(white_texture->bind_group);
		wgpuTextureViewRelease(white_texture->view);
		wgpuTextureRelease(white_texture->texture);
		delete white_texture;
		white_texture = nullptr;
	}

	auto release_pipeline = [](WGPURenderPipeline& pipeline) {
		if (pipeline)
			wgpuRenderPipelineRelease(pipeline);
		pipeline = nullptr;
	};
	release_pipeline(pipeline_main);
	release_pipeline(pipeline_main_clipped);
	release_pipeline(pipeline_mask_replace);
	release_pipeline(pipeline_mask_increment);
	release_pipeline(pipeline_gradient);
	release_pipeline(pipeline_gradient_clipped);
	release_pipeline(pipeline_creation);
	release_pipeline(pipeline_creation_clipped);
	release_pipeline(pipeline_passthrough_blend);
	release_pipeline(pipeline_passthrough_replace);
	release_pipeline(pipeline_opacity);
	release_pipeline(pipeline_blur);
	release_pipeline(pipeline_drop_shadow);
	release_pipeline(pipeline_color_matrix);
	release_pipeline(pipeline_blend_mask);

	if (uniform_buffer)
		wgpuBufferRelease(uniform_buffer);
	if (sampler)
		wgpuSamplerRelease(sampler);
	if (render_target_sampler)
		wgpuSamplerRelease(render_target_sampler);
	if (pipeline_layout)
		wgpuPipelineLayoutRelease(pipeline_layout);
	if (bind_group_layout)
		wgpuBindGroupLayoutRelease(bind_group_layout);

	uniform_buffer = nullptr;
	sampler = nullptr;
	render_target_sampler = nullptr;
	pipeline_layout = nullptr;
	bind_group_layout = nullptr;

	ReleasePendingResources();

	surface_target = {};
	root_target = {};
	current_target = nullptr;
	device = nullptr;
	queue = nullptr;
	target_format = WGPUTextureFormat_Undefined;
}
