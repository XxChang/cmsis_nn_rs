use core::any::TypeId;

use super::alloc::vec::Vec;
use crate::{
    pooling::{max_pool_s8, PoolParams},
    private::{cmsis_nn_activation, cmsis_nn_dims, cmsis_nn_pool_params, cmsis_nn_tile},
    tflite_adapter::schema_generated,
    Dims, NNContext,
};

pub struct MaxPoolParams {
    filter_dims: cmsis_nn_dims,
    input_dims: cmsis_nn_dims,
    output_dims: cmsis_nn_dims,
    pool_params: cmsis_nn_pool_params,
}

impl MaxPoolParams {
    pub fn new(
        op: &schema_generated::tflite::Operator<'_>,
        subgraph: &crate::tflite_adapter::subgraph::SubGraph,
    ) -> Option<Self> {
        if let Some(schema_params) = op.builtin_options_as_pool_2_doptions() {
            let stride_width = schema_params.stride_w();
            let stride_height = schema_params.stride_h();
            let filter_width = schema_params.filter_width();
            let filter_height = schema_params.filter_height();

            let inputs_array: Vec<i32> = op.inputs().iter().flatten().map(|x| x).collect();
            let outputs_array: Vec<i32> = op.outputs().iter().flatten().map(|x| x).collect();

            let input_index = inputs_array[0];
            let output_index = outputs_array[0];

            let tensors = subgraph.tensors();
            let input = &tensors[input_index as usize];
            let output = &tensors[output_index as usize];

            assert!(input.data_typ() == output.data_typ());

            // !todo support more type
            assert!(input.data_typ() == TypeId::of::<i8>());

            let depth = input.dims()[3];

            let filter_dims = cmsis_nn_dims {
                n: 1,
                h: filter_height,
                w: filter_width,
                c: 1,
            };

            let input_dims = cmsis_nn_dims {
                n: 1,
                h: input.dims()[1],
                w: input.dims()[2],
                c: depth,
            };

            let output_dims = cmsis_nn_dims {
                n: 1,
                h: output.dims()[1],
                w: output.dims()[2],
                c: depth,
            };

            let (act_min, act_max) = match schema_params.fused_activation_function().0 {
                x if x == schema_generated::tflite::ActivationFunctionType::NONE.0 => {
                    super::calculate_activation_range_quantized(output, x)
                }
                _ => panic!("Unsupported activation options"),
            };

            let input_height = input.dims()[1];
            let input_width = input.dims()[2];

            let out_height = output.dims()[1];
            let out_width = output.dims()[2];

            let padding_height =
                super::compute_padding(stride_height, 1, input_height, filter_height, out_height);
            let padding_width =
                super::compute_padding(stride_width, 1, input_width, filter_width, out_width);

            let pool_params = cmsis_nn_pool_params {
                stride: cmsis_nn_tile {
                    h: stride_height,
                    w: stride_width,
                },

                padding: cmsis_nn_tile {
                    h: padding_height,
                    w: padding_width,
                },

                activation: cmsis_nn_activation {
                    min: act_min,
                    max: act_max,
                },
            };

            Some(MaxPoolParams {
                input_dims,
                output_dims,
                filter_dims,
                pool_params,
            })
        } else {
            None
        }
    }

    pub fn eval(self, input_data: &[i8], output: &mut [i8]) -> Result<(), crate::Error> {
        let input_dims = Dims::new(
            self.input_dims.n,
            self.input_dims.h,
            self.input_dims.w,
            self.input_dims.c,
        );
        let output_dims = Dims::new(
            self.output_dims.n,
            self.output_dims.h,
            self.output_dims.w,
            self.output_dims.c,
        );
        let filter_dims = Dims::new(
            self.filter_dims.n,
            self.filter_dims.h,
            self.filter_dims.w,
            self.filter_dims.c,
        );
        let pool_params = PoolParams::new(
            (self.pool_params.stride.w, self.pool_params.stride.h),
            (self.pool_params.padding.w, self.pool_params.padding.h),
            (
                self.pool_params.activation.min,
                self.pool_params.activation.max,
            ),
        );
        defmt::debug!("pool params: {}", pool_params);
        defmt::debug!("input dims: {:?}", input_dims);
        defmt::debug!("filter dims: {:?}", filter_dims);
        defmt::debug!("output dims: {:?}", output_dims);

        let ctx = NNContext::default();
        max_pool_s8(
            &ctx,
            &pool_params,
            &input_dims,
            input_data,
            &filter_dims,
            &output_dims,
            output,
        )
    }
}
