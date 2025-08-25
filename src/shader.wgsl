struct Params {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    filtertype: u32,
    param: f32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>  src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

const PI: f32 = 3.141592653589793;

fn pack_rgba(v: vec4<f32>) -> u32 {
    let r: u32 = u32(clamp(v.r, 0.0, 1.0) * 255.0 + 0.5);
    let g: u32 = u32(clamp(v.g, 0.0, 1.0) * 255.0 + 0.5);
    let b: u32 = u32(clamp(v.b, 0.0, 1.0) * 255.0 + 0.5);
    let a: u32 = u32(clamp(v.a, 0.0, 1.0) * 255.0 + 0.5);
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

fn unpack_rgba(v: u32) -> vec4<f32> {
    let r: f32 = f32(v & 0xFFu) / 255.0;
    let g: f32 = f32((v >> 8u) & 0xFFu) / 255.0;
    let b: f32 = f32((v >> 16u) & 0xFFu) / 255.0;
    let a: f32 = f32((v >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

fn kernel_weight(x: f32, filtertype: u32, param: f32) -> f32 {
    switch filtertype {
        // Nearest
        case 0u: {
            return select(0.0, 1.0, abs(x) < 0.5);
        }
        // Triangle (Bilinear in 1D)
        case 1u: {
            return max(0.0, 1.0 - abs(x));
        }
        // Gaussian
        case 2u: {
            return exp(-0.5 * (x * x) / (param * param));
        }
        // Lanczos
        case 3u: {
            if x == 0.0 { return 1.0; }
            if abs(x) > param { return 0.0; }
            let pix = PI * x;
            return (sin(pix) / pix) * (sin(pix / param) / (pix / param));
        }
        // Box
        case 4u: {
            return 1.0;
        }
        default: {
            return 0.0;
        }
}
}

fn resample(xs: f32, ys: f32) -> vec4<f32> {
    var sum: vec4<f32> = vec4<f32>(0.0);
    var norm: f32 = 0.0;

    let radius: i32 = i32(ceil(params.param));

    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let sx: u32 = u32(clamp(i32(floor(xs)) + dx, 0, i32(params.in_width) - 1));
            let sy: u32 = u32(clamp(i32(floor(ys)) + dy, 0, i32(params.in_height) - 1));

            let w: f32 = kernel_weight(f32(dx) - fract(xs), params.filtertype, params.param) * kernel_weight(f32(dy) - fract(ys), params.filtertype, params.param);

            sum += unpack_rgba(src[sy * params.in_width + sx]) * w;
            norm += w;
        }
    }

    return sum / max(norm, 1e-6);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.out_width || id.y >= params.out_height {
        return;
    }

    let xs: f32 = (f32(id.x) + 0.5) * f32(params.in_width) / f32(params.out_width);
    let ys: f32 = (f32(id.y) + 0.5) * f32(params.in_height) / f32(params.out_height);

    let color: vec4<f32> = resample(xs, ys);
    let color2: vec4<f32> = vec4<f32>(f32(id.x) / f32(params.out_width), f32(id.y) / f32(params.out_height), 0.5, 1.0);
    let idx: u32 = id.y * params.out_width + id.x;
    dst[idx] = pack_rgba(color);
}
