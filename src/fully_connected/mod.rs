pub mod batch_matmul;
pub mod vector_sum;

use crate::{
    private::{arm_fully_connected_s4, cmsis_nn_activation, cmsis_nn_fc_params},
    NNContext, Result, StatusCode,
};

pub struct FcParams(cmsis_nn_fc_params);

impl FcParams {
    pub fn new(
        input_offset: i32,
        filter_offset: i32,
        output_offset: i32,
        range: (i32, i32),
    ) -> FcParams {
        FcParams(cmsis_nn_fc_params {
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

impl defmt::Format for FcParams {
    fn format(&self, fmt: defmt::Formatter) {
        defmt::write!(fmt, "FcParams {{");
        defmt::write!(fmt, "input_offset: {}, ", self.0.input_offset);
        defmt::write!(fmt, "filter_offset: {}, ", self.0.filter_offset);
        defmt::write!(fmt, "output_offset: {}, ", self.0.output_offset);
        defmt::write!(fmt, "activation: {{");
        defmt::write!(fmt, "min: {}, ", self.0.activation.min);
        defmt::write!(fmt, "max: {}", self.0.activation.max);
        defmt::write!(fmt, "}}");
        defmt::write!(fmt, "}}");
    }
}

impl AsRef<cmsis_nn_fc_params> for FcParams {
    fn as_ref(&self) -> &cmsis_nn_fc_params {
        &self.0
    }
}

pub fn fully_connected_wrapper_s8(
    ctx: &NNContext,
    fc_params: &FcParams,
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
    fc_params: &FcParams,
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
    fc_params: &FcParams,
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
    let bias_ptr = if bias.is_empty() {
        core::ptr::null()
    } else {
        bias.as_ptr()
    };

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
            bias_ptr,
            output_dims.as_ref(),
            output.as_mut_ptr(),
        )
    }
    .check_status()
}

pub fn fully_connected_s4(
    ctx: &NNContext,
    fc_params: &FcParams,
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

#[cfg(test)]
pub(crate) mod tests {
    const FC_PER_CH_IN_CH: i32 = 89;
    const FC_PER_CH_OUT_CH: i32 = 22;
    const FC_PER_CH_PER_CHANNEL_QUANT: bool = true;
    const FC_PER_CH_BATCH_SIZE: i32 = 1;
    const FC_PER_CH_OUT_ACTIVATION_MIN: i32 = -128;
    const FC_PER_CH_OUT_ACTIVATION_MAX: i32 = 127;
    const FC_PER_CH_INPUT_BATCHES: i32 = 1;
    const FC_PER_CH_INPUT_W: i32 = 1;
    const FC_PER_CH_INPUT_H: i32 = 1;
    const FC_PER_CH_DST_SIZE: i32 = 22;
    const FC_PER_CH_ACCUMULATION_DEPTH: i32 = 89;
    const FC_PER_CH_INPUT_OFFSET: i32 = 128;
    const FC_PER_CH_OUTPUT_OFFSET: i32 = 11;

    pub fn fc_per_ch_fully_connected_s8() {}
}
