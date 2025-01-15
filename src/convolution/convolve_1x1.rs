use crate::{NNContext, Result, StatusCode};

pub fn convolve_1x1_s4_fast(
    ctx: &NNContext,
    conv_params: &super::Config,
    per_channel_quant_params: &crate::PerChannelQuantParams,
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
        crate::private::arm_convolve_1x1_s4_fast(
            ctx.as_ref(),
            conv_params.as_ref(),
            per_channel_quant_params.as_ref(),
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

pub fn convolve_1x1_s4(
    ctx: &NNContext,
    conv_params: &super::Config,
    per_channel_quant_params: &crate::PerChannelQuantParams,
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
        crate::private::arm_convolve_1x1_s4(
            ctx.as_ref(),
            conv_params.as_ref(),
            per_channel_quant_params.as_ref(),
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

pub fn convolve_1x1_s8_fast(
    ctx: &NNContext,
    conv_params: &super::Config,
    per_channel_quant_params: &crate::PerChannelQuantParams,
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
        crate::private::arm_convolve_1x1_s8_fast(
            ctx.as_ref(),
            conv_params.as_ref(),
            per_channel_quant_params.as_ref(),
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

pub fn convolve_1x1_s8(
    ctx: &NNContext,
    conv_params: &super::Config,
    per_channel_quant_params: &crate::PerChannelQuantParams,
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
        crate::private::arm_convolve_1x1_s8(
            ctx.as_ref(),
            conv_params.as_ref(),
            per_channel_quant_params.as_ref(),
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
