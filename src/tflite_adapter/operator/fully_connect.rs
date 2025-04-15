use super::alloc::vec::Vec;
use crate::{
    fully_connected::{fully_connected_wrapper_s8, FcParams},
    private::{cmsis_nn_activation, cmsis_nn_fc_params},
    tflite_adapter::schema_generated,
    Dims, NNContext, QuantParams,
};
use core::any::TypeId;

pub struct FullyConnectedParams<'a> {
    accum_depth: i32,
    batches: i32,
    output_dims_count: usize,

    pub per_channel_multiplier: Vec<i32>,
    pub per_channel_shift: Vec<i32>,

    is_per_channel: bool,
    fc_params: cmsis_nn_fc_params,
    output_depth: i32,
    pub weights: &'a [i8],
    pub bias: &'a [i32],
}

impl<'b: 'a, 'a> FullyConnectedParams<'a> {
    pub fn new(
        op: &schema_generated::tflite::Operator<'_>,
        subgraph: &'b crate::tflite_adapter::subgraph::SubGraph,
    ) -> Option<Self> {
        if let Some(schema_params) = op.builtin_options_as_fully_connected_options() {
            let inputs_array: Vec<i32> = op.inputs().iter().flatten().map(|x| x).collect();
            let outputs_array: Vec<i32> = op.outputs().iter().flatten().map(|x| x).collect();

            let input_index = inputs_array[0];
            let weight_index = inputs_array[1];
            let bias_index = inputs_array[2];
            let output_index = outputs_array[0];

            let tensors = subgraph.tensors();
            let input = &tensors[input_index as usize];
            let filter = &tensors[weight_index as usize];
            let output = &tensors[output_index as usize];
            let bias = &tensors[bias_index as usize];

            assert!(input.data_typ() == output.data_typ());

            // !todo support more type
            assert!(input.data_typ() == TypeId::of::<i8>());

            let filter_dims_count = filter.dims().len();
            let output_dims_count = output.dims().len();

            assert!(output_dims_count >= 2);
            assert!(output_dims_count <= 4);

            let (act_min, act_max) = match schema_params.fused_activation_function().0 {
                x if x == schema_generated::tflite::ActivationFunctionType::NONE.0 => {
                    super::calculate_activation_range_quantized(output, x)
                }
                _ => panic!("Unsupported activation options"),
            };

            let fc_params = cmsis_nn_fc_params {
                input_offset: -input.zero_points()[0],
                filter_offset: -filter.zero_points()[0],
                output_offset: output.zero_points()[0],
                activation: cmsis_nn_activation {
                    min: act_min,
                    max: act_max,
                },
            };

            let is_per_channel = filter.scales().len() > 1;
            let num_channels = filter.dims()[0];

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

                let (multiplier, shift) = super::quantize_multiplier(effective_output_scale as f64);
                per_channel_multiplier.push(multiplier);
                per_channel_shift.push(shift);
            }

            defmt::debug!(
                "per channel multiplier: {:?}",
                per_channel_multiplier.as_slice()
            );
            defmt::debug!("per channel shift: {:?}", per_channel_shift.as_slice());

            Some(FullyConnectedParams {
                output_dims_count,

                accum_depth: filter.dims()[filter_dims_count - 1],
                output_depth: output.dims()[output_dims_count - 1],
                batches: super::flat_size_skip_dim(output.dims(), output_dims_count - 1),

                fc_params,
                is_per_channel,

                per_channel_multiplier,
                per_channel_shift,

                weights: filter.get_data_i8(),
                bias: bias.get_data_i32(),
            })
        } else {
            None
        }
    }

    pub fn eval(self, input_data: &[i8], output: &mut [i8]) -> Result<(), crate::Error> {
        if self.output_dims_count > 2 && self.accum_depth % 4 == 0 {
            todo!()
        } else {
            let fc_params = FcParams::new(
                self.fc_params.input_offset,
                self.fc_params.filter_offset,
                self.fc_params.output_offset,
                (self.fc_params.activation.min, self.fc_params.activation.max),
            );
            defmt::debug!("fc params: {}", fc_params);

            let fc_quant_params = QuantParams::new(
                self.per_channel_multiplier.as_slice(),
                self.per_channel_shift.as_slice(),
                if self.is_per_channel { 1 } else { 0 },
            );

            defmt::debug!(
                "per channel multiplier: {:?}",
                self.per_channel_multiplier.as_slice()
            );
            defmt::debug!("per channel shift: {:?}", self.per_channel_shift.as_slice());

            let fc_ctx = NNContext::default();

            let input_dims = Dims::new(self.batches, 1, 1, self.accum_depth);

            let filter_dims = Dims::new(self.accum_depth, 1, 1, self.output_depth);

            let bias_dims = Dims::new(1, 1, 1, self.output_depth);

            let output_dims = Dims::new(self.batches, 1, 1, self.output_depth);

            defmt::debug!("input dims: {:?}", input_dims);
            defmt::debug!("filter dims: {:?}", filter_dims);
            defmt::debug!("bias dims: {:?}", bias_dims);
            defmt::debug!("output dims: {:?}", output_dims);

            fully_connected_wrapper_s8(
                &fc_ctx,
                &fc_params,
                &fc_quant_params,
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
}
