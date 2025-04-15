use crate::{
    private::{cmsis_nn_activation, cmsis_nn_conv_params, cmsis_nn_tile},
    Dims, NNContext, Result, StatusCode,
};

pub struct ConvParams(cmsis_nn_conv_params);

impl defmt::Format for ConvParams {
    fn format(&self, fmt: defmt::Formatter) {
        let ConvParams(params) = self;
        defmt::write!(
            fmt,
            "ConvParams(input_offset: {}, output_offset: {}, stride: {:?}, padding: {:?}, dilation: {:?}, activation: {:?})",
            params.input_offset,
            params.output_offset,
            (params.stride.w, params.stride.h),
            (params.padding.w, params.padding.h),
            (params.dilation.w, params.dilation.h),
            (params.activation.min, params.activation.max),
        );
    }
}

impl ConvParams {
    pub fn new(
        input_offset: i32,
        output_offset: i32,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        range: (i32, i32),
    ) -> ConvParams {
        ConvParams(cmsis_nn_conv_params {
            input_offset,
            output_offset,
            stride: cmsis_nn_tile {
                w: stride.0,
                h: stride.1,
            },
            padding: cmsis_nn_tile {
                w: padding.0,
                h: padding.1,
            },
            dilation: cmsis_nn_tile {
                w: dilation.0,
                h: dilation.1,
            },
            activation: cmsis_nn_activation {
                min: range.0,
                max: range.1,
            },
        })
    }
}

impl AsRef<cmsis_nn_conv_params> for ConvParams {
    fn as_ref(&self) -> &cmsis_nn_conv_params {
        &self.0
    }
}

pub fn convolve_wrapper_s8(
    ctx: &NNContext,
    conv_params: &ConvParams,
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
    let bias_ptr = if bias_data.is_empty() {
        core::ptr::null()
    } else {
        bias_data.as_ptr()
    };

    unsafe {
        crate::private::arm_convolve_wrapper_s8(
            ctx.as_ref(),
            conv_params.as_ref(),
            per_channel_quant_params.as_ref(),
            input_dims.as_ref(),
            input_data.as_ptr(),
            filter_dims.as_ref(),
            filter_data.as_ptr(),
            bias_dims.as_ref(),
            bias_ptr,
            output_dims.as_ref(),
            output_data.as_mut_ptr(),
        )
    }
    .check_status()
}

pub fn convolve_wrapper_s8_get_buffer_size(
    conv_params: &ConvParams,
    input_dims: &Dims,
    filter_dims: &Dims,
    output_dims: &Dims,
) -> i32 {
    unsafe {
        crate::private::arm_convolve_wrapper_s8_get_buffer_size(
            conv_params.as_ref(),
            input_dims.as_ref(),
            filter_dims.as_ref(),
            output_dims.as_ref(),
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::PerChannelQuantParams;
    use crate::{test_utils::validate, Dims};

    const BASIC_OUT_CH: i32 = 1;
    const BASIC_IN_CH: i32 = 1;
    const BASIC_INPUT_W: i32 = 5;
    const BASIC_INPUT_H: i32 = 8;
    const BASIC_DST_SIZE: i32 = 20;
    // const BASIC_INPUT_SIZE: i32 = 40;
    const BASIC_OUT_ACTIVATION_MIN: i32 = -128;
    const BASIC_OUT_ACTIVATION_MAX: i32 = 127;
    const BASIC_INPUT_BATCHES: i32 = 1;
    const BASIC_FILTER_X: i32 = 2;
    const BASIC_FILTER_Y: i32 = 4;
    const BASIC_STRIDE_X: i32 = 1;
    const BASIC_STRIDE_Y: i32 = 1;
    const BASIC_PAD_X: i32 = 0;
    const BASIC_PAD_Y: i32 = 0;
    const BASIC_OUTPUT_W: i32 = 4;
    const BASIC_OUTPUT_H: i32 = 5;
    const BASIC_INPUT_OFFSET: i32 = 128;
    const BASIC_OUTPUT_OFFSET: i32 = 127;
    const BASIC_DILATION_X: i32 = 1;
    const BASIC_DILATION_Y: i32 = 1;

    const BASIC_BIASES: [i32; 1] = [6388];
    const BASIC_INPUT: [i8; 40] = [
        73, -88, -95, 57, 106, 13, 34, -103, 86, 12, 107, 37, -4, -22, 16, -87, 4, -11, -21, 52,
        41, -122, 90, 124, -62, -23, 103, 66, 68, 94, -93, 89, -4, 68, -89, -66, 3, 4, -108, 63,
    ];
    const BASIC_OUTPUT_REF: [i8; 20] = [
        -11, 37, 68, -53, -8, -47, -1, -6, 29, -86, -34, 27, -40, 34, -71, 4, -72, 21, -14, -35,
    ];
    const BASIC_WEIGHTS: [i8; 8] = [-72, 32, -107, -50, 81, -114, -7, -127];

    const BASIC_OUTPUT_MULT: [i32; 1] = [1625013239];
    const BASIC_OUTPUT_SHIFT: [i32; 1] = [-8];

    pub fn basic_convolve_s8() {
        let mut output = [0; BASIC_DST_SIZE as _];

        let bias_data = &BASIC_BIASES;
        let kernel_data = &BASIC_WEIGHTS;
        let input_data = &BASIC_INPUT;
        let output_ref = &BASIC_OUTPUT_REF;
        let output_ref_size = BASIC_DST_SIZE;
        let quant_params = PerChannelQuantParams::new(&BASIC_OUTPUT_MULT, &BASIC_OUTPUT_SHIFT);
        let input_dims = Dims::new(
            BASIC_INPUT_BATCHES,
            BASIC_INPUT_H,
            BASIC_INPUT_W,
            BASIC_IN_CH,
        );
        let filter_dims = Dims::new(1, BASIC_FILTER_Y, BASIC_FILTER_X, BASIC_IN_CH);
        let output_dims = Dims::new(1, BASIC_OUTPUT_H, BASIC_OUTPUT_W, BASIC_OUT_CH);
        let bias_dims = Dims::new(1, 1, 1, 1);

        let conv_params = ConvParams::new(
            BASIC_INPUT_OFFSET,
            BASIC_OUTPUT_OFFSET,
            (BASIC_STRIDE_X, BASIC_STRIDE_Y),
            (BASIC_PAD_X, BASIC_PAD_Y),
            (BASIC_DILATION_X, BASIC_DILATION_Y),
            (BASIC_OUT_ACTIVATION_MIN, BASIC_OUT_ACTIVATION_MAX),
        );

        let buf_size = convolve_wrapper_s8_get_buffer_size(
            &conv_params,
            &input_dims,
            &filter_dims,
            &output_dims,
        );

        let ctx = unsafe {
            NNContext::new_from_slice(&mut crate::test_utils::MEMORY[0..buf_size as usize])
        };

        assert!(convolve_wrapper_s8(
            &ctx,
            &conv_params,
            &quant_params,
            &input_dims,
            input_data,
            &filter_dims,
            kernel_data,
            &bias_dims,
            bias_data,
            &output_dims,
            &mut output,
        )
        .is_ok());

        assert!(validate(
            output.as_ptr(),
            output_ref.as_ptr(),
            output_ref_size as usize
        ));
    }

    const STRIDE2PAD1_OUT_CH: i32 = 1;
    const STRIDE2PAD1_IN_CH: i32 = 1;
    const STRIDE2PAD1_INPUT_W: i32 = 7;
    const STRIDE2PAD1_INPUT_H: i32 = 7;
    const STRIDE2PAD1_DST_SIZE: i32 = 16;
    const STRIDE2PAD1_OUT_ACTIVATION_MIN: i32 = -128;
    const STRIDE2PAD1_OUT_ACTIVATION_MAX: i32 = 127;
    const STRIDE2PAD1_INPUT_BATCHES: i32 = 1;
    const STRIDE2PAD1_FILTER_X: i32 = 3;
    const STRIDE2PAD1_FILTER_Y: i32 = 3;
    const STRIDE2PAD1_STRIDE_X: i32 = 2;
    const STRIDE2PAD1_STRIDE_Y: i32 = 2;
    const STRIDE2PAD1_PAD_X: i32 = 1;
    const STRIDE2PAD1_PAD_Y: i32 = 1;
    const STRIDE2PAD1_OUTPUT_W: i32 = 4;
    const STRIDE2PAD1_OUTPUT_H: i32 = 4;
    const STRIDE2PAD1_INPUT_OFFSET: i32 = 128;
    const STRIDE2PAD1_OUTPUT_OFFSET: i32 = -20;
    const STRIDE2PAD1_DILATION_X: i32 = 1;
    const STRIDE2PAD1_DILATION_Y: i32 = 1;

    const STRIDE2PAD1_BIASES: [i32; 1] = [-9794];
    const STRIDE2PAD1_WEIGHTS: [i8; 9] = [-54, 57, -19, -127, 87, 70, 74, -110, 66];
    const STRIDE2PAD1_INPUT: [i8; 49] = [
        -91, -30, -57, -76, 32, -13, 14, -96, 108, -4, 41, 48, 107, -68, -101, 30, 95, 95, 91, -66,
        -80, 114, -49, 7, -67, -35, -1, -88, -77, -56, -103, 5, -39, -118, -24, -32, 67, 11, 38,
        -16, -124, 44, -46, -92, -24, 108, 80, -29, -3,
    ];
    const STRIDE2PAD1_OUTPUT_REF: [i8; 16] = [
        26, -11, 33, -25, -96, -52, -78, -86, 33, -2, -88, -113, -14, 0, -84, -27,
    ];

    const STRIDE2PAD1_OUTPUT_MULT: [i32; 1] = [2033801520];
    const STRIDE2PAD1_OUTPUT_SHIFT: [i32; 1] = [-8];

    pub fn stride2pad1_convolve_s8() {
        let mut output = [0i8; STRIDE2PAD1_DST_SIZE as _];
        let bias_data = &STRIDE2PAD1_BIASES;
        let kernel_data = &STRIDE2PAD1_WEIGHTS;
        let input_data = &STRIDE2PAD1_INPUT;
        let output_ref = &STRIDE2PAD1_OUTPUT_REF;
        let output_ref_size = STRIDE2PAD1_DST_SIZE;

        let bias_dims = Dims::new(1, 1, 1, 1);

        let input_dims = Dims::new(
            STRIDE2PAD1_INPUT_BATCHES,
            STRIDE2PAD1_INPUT_H,
            STRIDE2PAD1_INPUT_W,
            STRIDE2PAD1_IN_CH,
        );
        let filter_dims = Dims::new(
            1,
            STRIDE2PAD1_FILTER_Y,
            STRIDE2PAD1_FILTER_X,
            STRIDE2PAD1_IN_CH,
        );
        let output_dims = Dims::new(
            1,
            STRIDE2PAD1_OUTPUT_H,
            STRIDE2PAD1_OUTPUT_W,
            STRIDE2PAD1_OUT_CH,
        );

        let conv_params = ConvParams::new(
            STRIDE2PAD1_INPUT_OFFSET,
            STRIDE2PAD1_OUTPUT_OFFSET,
            (STRIDE2PAD1_STRIDE_X, STRIDE2PAD1_STRIDE_Y),
            (STRIDE2PAD1_PAD_X, STRIDE2PAD1_PAD_Y),
            (STRIDE2PAD1_DILATION_X, STRIDE2PAD1_DILATION_Y),
            (
                STRIDE2PAD1_OUT_ACTIVATION_MIN,
                STRIDE2PAD1_OUT_ACTIVATION_MAX,
            ),
        );
        let quant_params =
            PerChannelQuantParams::new(&STRIDE2PAD1_OUTPUT_MULT, &STRIDE2PAD1_OUTPUT_SHIFT);

        let buf_size = convolve_wrapper_s8_get_buffer_size(
            &conv_params,
            &input_dims,
            &filter_dims,
            &output_dims,
        );
        let ctx = unsafe {
            NNContext::new_from_slice(&mut crate::test_utils::MEMORY[0..buf_size as usize])
        };

        assert!(convolve_wrapper_s8(
            &ctx,
            &conv_params,
            &quant_params,
            &input_dims,
            input_data,
            &filter_dims,
            kernel_data,
            &bias_dims,
            bias_data,
            &output_dims,
            &mut output
        )
        .is_ok());

        assert!(validate(
            output.as_ptr(),
            output_ref.as_ptr(),
            output_ref_size as usize
        ));
    }
}
