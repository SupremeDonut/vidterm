use wgpu::util::DeviceExt;

#[derive(Clone, Copy)]
pub enum FilterType {
    Nearest = 0,
    Bilinear = 1,
    Gaussian = 2,
    Lanczos = 3,
    Box = 4,
}

pub struct Compute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl Compute {
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        Self {
            device,
            queue,
            pipeline,
        }
    }

    pub fn resample_gpu(
        &self,
        frame: &Vec<u8>,
        input_size: (usize, usize),
        output_size: (usize, usize),
        filtering: FilterType,
        strength: f32,
    ) -> Vec<u8> {
        let (in_w, in_h) = (input_size.0 as u32, input_size.1 as u32);
        let (out_w, out_h) = (output_size.0 as u32, output_size.1 as u32);
        // let in_bytes = (in_w * in_h * 4) as u64;
        let out_bytes = (out_w * out_h * 4) as u64;

        // src/dst storage buffers
        let src_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("src"),
                contents: frame,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let dst_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dst"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            in_w: u32,
            in_h: u32,
            out_w: u32,
            out_h: u32,
            filter: u32,
            param: f32,
        }
        let params = Params {
            in_w,
            in_h,
            out_w,
            out_h,
            filter: filtering as u32,
            param: strength,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // bind group
        let bgl = self.pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst_buf.as_entire_binding(),
                },
            ],
        });

        // dispatch (8x8)
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((out_w + 7) / 8, (out_h + 7) / 8, 1);
        }

        // readback
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&dst_buf, 0, &read, 0, out_bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait).ok();
        let data = slice.get_mapped_range().to_vec();
        read.unmap();
        data
    }
}
