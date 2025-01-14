pub mod batch_matmul;
pub mod vector_sum;

use crate::{
    private::{arm_fully_connected_s4, cmsis_nn_activation, cmsis_nn_fc_params},
    NNContext, Result, StatusCode,
};

pub struct Config(cmsis_nn_fc_params);

impl Config {
    pub fn new(
        input_offset: i32,
        filter_offset: i32,
        output_offset: i32,
        range: (i32, i32),
    ) -> Config {
        Config(cmsis_nn_fc_params {
            input_offset,
            filter_offset,
            output_offset,
            activation: cmsis_nn_activation {
                min: range.0,
                max: range.1,
            },
        })
    }
}

impl AsRef<cmsis_nn_fc_params> for Config {
    fn as_ref(&self) -> &cmsis_nn_fc_params {
        &self.0
    }
}

pub fn fully_connected_wrapper_s8(
    ctx: &NNContext,
    fc_params: &Config,
    quant_params: &crate::QuantParams,
    input_dims: &crate::Dims,
    input_data: &[i8],
    filter_dims: &crate::Dims,
    filter_data: &[i8],
    bias_dims: &crate::Dims,
    bias_data: &[i32],
    output_dims: &crate::Dims,
    output_data: &mut [i8],
) -> Result<()> {
    unsafe {
        crate::private::arm_fully_connected_wrapper_s8(
            ctx.as_ref(),
            fc_params.as_ref(),
            quant_params.as_ref(),
            input_dims.as_ref(),
            input_data.as_ptr(),
            filter_dims.as_ref(),
            filter_data.as_ptr(),
            bias_dims.as_ref(),
            bias_data.as_ptr(),
            output_dims.as_ref(),
            output_data.as_mut_ptr(),
        )
    }
    .check_status()
}

pub fn fully_connected_s16(
    ctx: &NNContext,
    fc_params: &Config,
    quant_params: &crate::PerTensorQuantParams,
    input_dims: &crate::Dims,
    input: &[i16],
    filter_dims: &crate::Dims,
    kernel: &[i8],
    bias_dims: &crate::Dims,
    bias_data: &[i64],
    output_dims: &crate::Dims,
    output_data: &mut [i16],
) -> Result<()> {
    unsafe {
        crate::private::arm_fully_connected_s16(
            ctx.as_ref(),
            fc_params.as_ref(),
            quant_params.as_ref(),
            input_dims.as_ref(),
            input.as_ptr(),
            filter_dims.as_ref(),
            kernel.as_ptr(),
            bias_dims.as_ref(),
            bias_data.as_ptr(),
            output_dims.as_ref(),
            output_data.as_mut_ptr(),
        )
    }
    .check_status()
}

pub fn fully_connected_s8(
    ctx: &NNContext,
    fc_params: &Config,
    quant_params: &crate::PerTensorQuantParams,
    input_dims: &crate::Dims,
    input: &[i8],
    filter_dims: &crate::Dims,
    kernel: &[i8],
    bias_dims: &crate::Dims,
    bias: &[i32],
    output_dims: &crate::Dims,
    output: &mut [i8],
) -> Result<()> {
    unsafe {
        crate::private::arm_fully_connected_s8(
            ctx.as_ref(),
            fc_params.as_ref(),
            quant_params.as_ref(),
            input_dims.as_ref(),
            input.as_ptr(),
            filter_dims.as_ref(),
            kernel.as_ptr(),
            bias_dims.as_ref(),
            bias.as_ptr(),
            output_dims.as_ref(),
            output.as_mut_ptr(),
        )
    }
    .check_status()
}

pub fn fully_connected_s4(
    ctx: &NNContext,
    fc_params: &Config,
    quant_params: &crate::PerTensorQuantParams,
    input_dims: &crate::Dims,
    input: &[i8],
    filter_dims: &crate::Dims,
    kernel: &[i8],
    bias_dims: &crate::Dims,
    bias: &[i32],
    output_dims: &crate::Dims,
    output: &mut [i8],
) -> Result<()> {
    unsafe {
        arm_fully_connected_s4(
            ctx.as_ref(),
            fc_params.as_ref(),
            quant_params.as_ref(),
            input_dims.as_ref(),
            input.as_ptr(),
            filter_dims.as_ref(),
            kernel.as_ptr(),
            bias_dims.as_ref(),
            bias.as_ptr(),
            output_dims.as_ref(),
            output.as_mut_ptr(),
        )
    }
    .check_status()
}
