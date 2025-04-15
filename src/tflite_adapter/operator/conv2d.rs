use core::{any::TypeId, i8};

use super::alloc;
use crate::{
    convolution::{convolve_wrapper_s8, convolve_wrapper_s8_get_buffer_size, ConvParams},
    private::{cmsis_nn_activation, cmsis_nn_conv_params, cmsis_nn_dims, cmsis_nn_tile},
    tflite_adapter::schema_generated,
    Dims, NNContext, PerChannelQuantParams,
};

use alloc::vec::Vec;

pub struct Conv2DParams<'a> {
    pub stride_width: i32,
    pub stride_height: i32,
    pub dilation_width_factor: i32,
    pub dilation_height_factor: i32,

    pub input_dims: cmsis_nn_dims,
    pub filter_dims: cmsis_nn_dims,
    pub output_dims: cmsis_nn_dims,

    pub conv_params: cmsis_nn_conv_params,

    pub per_channel_multiplier: Vec<i32>,
    pub per_channel_shift: Vec<i32>,

    pub weights: &'a [i8],
    pub bias: &'a [i32],
}

pub static mut MEMORY: [i8; 10 * 1024] = [0; 10 * 1024];

impl<'b: 'a, 'a> Conv2DParams<'a> {
    pub fn new(
        op: &schema_generated::tflite::Operator<'_>,
        subgraph: &'b crate::tflite_adapter::subgraph::SubGraph,
    ) -> Option<Self> {
        if let Some(schema_params) = op.builtin_options_as_conv_2_doptions() {
            let stride_width = schema_params.stride_w();
            let stride_height = schema_params.stride_h();

            let dilation_width_factor = schema_params.dilation_w_factor();
            let dilation_height_factor = schema_params.dilation_h_factor();

            let inputs_array: Vec<i32> = op.inputs().iter().flatten().map(|x| x).collect();
            let outputs_array: Vec<i32> = op.outputs().iter().flatten().map(|x| x).collect();

            let input_index = inputs_array[0];
            let weight_index = inputs_array[1];
            let bias_index = inputs_array[2];
            let output_index = outputs_array[0];
            defmt::debug!("input_index: {}", input_index);
            defmt::debug!("weight_index: {}", weight_index);
            defmt::debug!("bias_index: {}", bias_index);
            defmt::debug!("output_index: {}", output_index);

            let tensors = subgraph.tensors();
            let input = &tensors[input_index as usize];
            let filter = &tensors[weight_index as usize];
            let output = &tensors[output_index as usize];
            let bias = &tensors[bias_index as usize];

            assert!(input.data_typ() == output.data_typ());

            // !todo support more type
            assert!(input.data_typ() == TypeId::of::<i8>());

            assert_eq!(input.dims().len(), 4);
            assert_eq!(filter.dims().len(), 4);
            assert_eq!(output.dims().len(), 4);
            assert_eq!(input.dims()[0], output.dims()[0]);

            assert!(filter.dims()[3] > 0);
            assert_eq!(input.dims()[3] % filter.dims()[3], 0);

            // Output channels should be an even multiple of the number of groups
            let groups = input.dims()[3] / filter.dims()[3];
            assert_eq!(output.dims()[3] % groups, 0);

            let (act_min, act_max) = match schema_params.fused_activation_function().0 {
                x if x == schema_generated::tflite::ActivationFunctionType::RELU.0 => {
                    super::calculate_activation_range_quantized(output, x)
                }
                _ => panic!("Unsupported activation options"),
            };
            // if bias.is_data_stored_in() {
            //     assert_eq!(bias.dims().len(), 4);
            //     defmt::info!("bias.dims(): {:?}", bias.dims());
            // }

            let out_height = output.dims()[1];
            let out_width = output.dims()[2];

            let input_height = input.dims()[1];
            let input_width = input.dims()[2];

            let padding_height = super::compute_padding(
                stride_height,
                dilation_height_factor,
                input_height,
                filter.dims()[1],
                out_height,
            );
            let padding_width = super::compute_padding(
                stride_width,
                dilation_width_factor,
                input_width,
                filter.dims()[2],
                out_width,
            );

            let num_channels = filter.dims()[0];

            let is_per_channel = filter.scales().len() > 1;

            let mut per_channel_multiplier = Vec::new();
            let mut per_channel_shift = Vec::new();

            for i in 0..num_channels {
                let filter_scale = if is_per_channel {
                    filter.scales()[i as usize]
                } else {
                    filter.scales()[0 as usize]
                };

                let input_scale = input.scales()[0] as f64;
                let output_scale = output.scales()[0] as f64;
                let effective_output_scale = input_scale * filter_scale as f64 / output_scale;

                let (multiplier, shift) = super::quantize_multiplier(effective_output_scale);
                per_channel_multiplier.push(multiplier);
                per_channel_shift.push(shift);
            }

            let conv_params = cmsis_nn_conv_params {
                input_offset: -1 * input.zero_points()[0],
                output_offset: output.zero_points()[0],
                stride: cmsis_nn_tile {
                    w: stride_width,
                    h: stride_height,
                },
                dilation: cmsis_nn_tile {
                    w: dilation_width_factor,
                    h: dilation_height_factor,
                },
                padding: cmsis_nn_tile {
                    w: padding_width,
                    h: padding_height,
                },
                activation: cmsis_nn_activation {
                    min: act_min,
                    max: act_max,
                },
            };

            Some(Conv2DParams {
                stride_width,
                stride_height,

                dilation_width_factor,
                dilation_height_factor,

                input_dims: cmsis_nn_dims {
                    n: input.dims()[0],
                    h: input.dims()[1],
                    w: input.dims()[2],
                    c: input.dims()[3],
                },

                filter_dims: cmsis_nn_dims {
                    n: 1,
                    h: filter.dims()[1],
                    w: filter.dims()[2],
                    c: filter.dims()[3],
                },

                output_dims: cmsis_nn_dims {
                    n: output.dims()[0],
                    h: output.dims()[1],
                    w: output.dims()[2],
                    c: output.dims()[3],
                },

                per_channel_multiplier,
                per_channel_shift,

                conv_params,

                weights: filter.get_data_i8(),
                bias: bias.get_data_i32(),
            })
        } else {
            None
        }
    }

    pub fn eval(self, input_data: &[i8], output: &mut [i8]) -> Result<(), crate::Error> {
        let conv_params = ConvParams::new(
            self.conv_params.input_offset,
            self.conv_params.output_offset,
            (self.stride_width, self.stride_height),
            (self.conv_params.padding.w, self.conv_params.padding.h),
            (self.conv_params.dilation.w, self.conv_params.dilation.h),
            (
                self.conv_params.activation.min,
                self.conv_params.activation.max,
            ),
        );

        defmt::debug!("conv params: {}", conv_params);
        let quant_params = PerChannelQuantParams::new(
            self.per_channel_multiplier.as_slice(),
            self.per_channel_shift.as_slice(),
        );
        defmt::debug!(
            "per channel multiplier: {:?}",
            self.per_channel_multiplier.as_slice()
        );
        defmt::debug!("per channel shift: {:?}", self.per_channel_shift.as_slice());

        let input_dims = Dims::new(
            self.input_dims.n,
            self.input_dims.h,
            self.input_dims.w,
            self.input_dims.c,
        );
        defmt::debug!("input_dims {:?}", input_dims);
        let filter_dims = Dims::new(
            self.filter_dims.n,
            self.filter_dims.h,
            self.filter_dims.w,
            self.filter_dims.c,
        );
        defmt::debug!("filter_dims {:?}", filter_dims);
        let output_dims = Dims::new(
            self.output_dims.n,
            self.output_dims.h,
            self.output_dims.w,
            self.output_dims.c,
        );
        defmt::debug!("output_dims {:?}", output_dims);
        let bias_dims = Dims::new(1, 1, 1, self.output_dims.c);
        defmt::debug!("bias_dims {:?}", bias_dims);
        let buf_size = convolve_wrapper_s8_get_buffer_size(
            &conv_params,
            &input_dims,
            &filter_dims,
            &output_dims,
        );

        let mut ctx = unsafe { NNContext::new_from_slice(&mut MEMORY[0..buf_size as usize]) };
        ctx.fill_zero();

        defmt::debug!("weights {:?}", self.weights);
        defmt::debug!("bias {:?}", self.bias);

        convolve_wrapper_s8(
            &ctx,
            &conv_params,
            &quant_params,
            &input_dims,
            input_data,
            &filter_dims,
            self.weights,
            &bias_dims,
            self.bias,
            &output_dims,
            output,
        )
    }
}
