use crate::tape::{Tape, Tensor};
use anyhow::Result;
use bytemuck;
use pollster::FutureExt;
use std::sync::mpsc::channel;
use std::sync::{Arc, OnceLock};
use wgpu::InstanceDescriptor;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Input {
    dims: [u32; 4],
    strides: [u32; 4],
    is_contiguous: u32,
    rank: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    a: Input,
    b: Input,
    k: u32,
    _padding: [u32; 3],
}

impl Uniforms {
    fn new(a: Tensor, b: Tensor) -> Self {
        let k = a.shape.get_dim(-1) as u32;

        let a = Input {
            dims: a.shape.dims.0.map(|x| x as u32),
            strides: a.shape.strides.0.map(|x| x as u32),
            is_contiguous: a.is_contiguous as u32,
            rank: a.shape.rank() as u32,
            _padding: [0; 2],
        };

        let b = Input {
            dims: b.shape.dims.0.map(|x| x as u32),
            strides: b.shape.strides.0.map(|x| x as u32),
            is_contiguous: b.is_contiguous as u32,
            rank: b.shape.rank() as u32,
            _padding: [0; 2],
        };

        Self {
            a,
            b,
            k,
            _padding: [0; 3],
        }
    }
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl Gpu {
    async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&Default::default()).await?;
        let (device, queue) = adapter.request_device(&Default::default()).await?;
        let shader = device.create_shader_module(wgpu::include_wgsl!("bmm.wgsl"));

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bmm"),
            layout: None,
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    pub fn get() -> Arc<Self> {
        static GPU: OnceLock<Arc<Gpu>> = OnceLock::new();

        let gpu = GPU.get_or_init(|| {
            let gpu = Self::new().block_on().unwrap();
            Arc::new(gpu)
        });

        gpu.clone()
    }

    // Expects a and b to already have been broadcast correctly
    pub async fn bmm(&self, tape: &mut Tape, a: usize, b: usize) -> Result<()> {
        let a = tape.weights[a];
        let b = tape.weights[b];
        let a_data = &tape.data[a.offset..a.offset + a.shape.span()];
        let b_data = &tape.data[b.offset..b.offset + b.shape.span()];

        // Create GPU buffers

        let uniforms = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&Uniforms::new(a, b)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let a_input = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("a_input"),
            contents: bytemuck::cast_slice(a_data),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let b_input = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("b_input"),
            contents: bytemuck::cast_slice(b_data),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let n_batches = a.shape.batches().product();
        let m = a.shape.get_dim(-2);
        let n = b.shape.get_dim(-1);
        let output_len = n_batches * m * n;
        let output_size = output_len as u64 * 4;

        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let temp = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temp"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Bind buffers

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        // GPU work

        let mut encoder = self.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let num_dispatches = output_len.div_ceil(64) as u32;
        pass.dispatch_workgroups(num_dispatches, 1, 1);
        drop(pass);

        encoder.copy_buffer_to_buffer(&output, 0, &temp, 0, output.size());
        self.queue.submit([encoder.finish()]);

        // Copy output to the CPU

        let (sender, receiver) = channel();

        temp.map_async(wgpu::MapMode::Read, .., move |result| {
            sender.send(result).unwrap()
        });

        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver.recv()??; // check that mapping was successful

        let mapped = temp.get_mapped_range(..)?;
        let data: &[f32] = bytemuck::cast_slice(&mapped);
        tape.data.extend_from_slice(data);

        Ok(())
    }
}
