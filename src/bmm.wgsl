struct Input {
    dims: vec4<u32>,
    strides: vec4<u32>,
    is_contiguous: u32,
    rank: u32,
}

struct Uniforms {
    a: Input,
    b: Input,
    k: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let total = arrayLength(&output);
    let stride = num_workgroups.x * 64;

    var gid = global_invocation_id.x;
    while (gid < total) {
        let k = uniforms.k;
        let m = uniforms.a.dims[uniforms.a.rank - 2];
        let n = uniforms.b.dims[uniforms.b.rank - 1];
        let batch = gid / (m * n);
        let row = (gid % (m * n)) / n;
        let col = gid % n;

        let a_start = batch * m * k;
        let b_start = batch * n * k;

        var dot = 0.0;
        for (var i = 0u; i < k; i++) {
            let a_i = get_index(uniforms.a, a_start + row * k + i);
            let b_i = get_index(uniforms.b, b_start + i * n + col);

            dot += a[a_i] * b[b_i];
        }

        output[gid] = dot;
        gid += stride;
    }
}

fn get_index(input: Input, i: u32) -> u32 {
    if input.is_contiguous == 1 {
        return i;
    } else {
        let index = unravel_index(input, i);

        var dot = 0u;
        for (var j = 0; j < 4; j++) {
            dot += index[j] * input.strides[j];
        }

        return dot;
    }
}

fn unravel_index(input: Input, i: u32) -> vec4<u32> {
    var index = vec4<u32>(0u);

    var j = i;
    for (var k = 0u; k < input.rank; k++) {
        let idx = input.rank - 1u - k;
        let dim = input.dims[idx];

        index[idx] = j % dim;
        j = j / dim;
    }

    return index;
}
