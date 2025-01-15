use crate::{
    private::{cmsis_nn_activation, cmsis_nn_pool_params, cmsis_nn_tile},
    Dims, NNContext, StatusCode,
};
use crate::{test_length, Error, Result};

pub struct Config(cmsis_nn_pool_params);

impl Config {
    pub fn new(stride: (i32, i32), padding: (i32, i32), range: (i32, i32)) -> Config {
        Config(cmsis_nn_pool_params {
            stride: cmsis_nn_tile {
                w: stride.0,
                h: stride.1,
            },
            padding: cmsis_nn_tile {
                w: padding.0,
                h: padding.1,
            },
            activation: cmsis_nn_activation {
                min: range.0,
                max: range.1,
            },
        })
    }
}

impl AsRef<cmsis_nn_pool_params> for Config {
    fn as_ref(&self) -> &cmsis_nn_pool_params {
        &self.0
    }
}

pub fn max_pool_s16(
    ctx: &NNContext,
    pool_params: &Config,
    input_dims: &Dims,
    src: &[i16],
    filter_dims: &Dims,
    output_dims: &Dims,
    dst: &mut [i16],
) -> Result<()> {
    test_length!(
        src,
        input_dims.0.n * input_dims.0.h * input_dims.0.w * input_dims.0.c
    )?;
    test_length!(
        dst,
        output_dims.0.n * output_dims.0.h * output_dims.0.w * output_dims.0.c
    )?;

    let status = unsafe {
        crate::private::arm_max_pool_s16(
            ctx.as_ref(),
            pool_params.as_ref(),
            input_dims.as_ref(),
            src.as_ptr() as *const _,
            filter_dims.as_ref(),
            output_dims.as_ref(),
            dst.as_mut_ptr() as *mut _,
        )
    };

    status.check_status()
}

pub fn max_pool_s8(
    ctx: &NNContext,
    pool_params: &Config,
    input_dims: &Dims,
    src: &[i8],
    filter_dims: &Dims,
    output_dims: &Dims,
    dst: &mut [i8],
) -> Result<()> {
    test_length!(
        src,
        input_dims.0.n * input_dims.0.h * input_dims.0.w * input_dims.0.c
    )?;
    test_length!(
        dst,
        output_dims.0.n * output_dims.0.h * output_dims.0.w * output_dims.0.c
    )?;

    let status = unsafe {
        crate::private::arm_max_pool_s8(
            ctx.as_ref(),
            pool_params.as_ref(),
            input_dims.as_ref(),
            src.as_ptr() as *const _,
            filter_dims.as_ref(),
            output_dims.as_ref(),
            dst.as_mut_ptr() as *mut _,
        )
    };

    status.check_status()
}

pub fn avgpool_s8(
    ctx: &NNContext,
    pool_params: &Config,
    input_dims: &Dims,
    src: &[i8],
    filter_dims: &Dims,
    output_dims: &Dims,
    dst: &mut [i8],
) -> Result<()> {
    test_length!(
        src,
        input_dims.0.n * input_dims.0.h * input_dims.0.w * input_dims.0.c
    )?;
    test_length!(
        dst,
        output_dims.0.n * output_dims.0.h * output_dims.0.w * output_dims.0.c
    )?;

    let status = unsafe {
        crate::private::arm_avgpool_s8(
            ctx.as_ref(),
            pool_params.as_ref(),
            input_dims.as_ref(),
            src.as_ptr() as *const _,
            filter_dims.as_ref(),
            output_dims.as_ref(),
            dst.as_mut_ptr() as *mut _,
        )
    };

    status.check_status()
}

pub fn avgpool_s16(
    ctx: &NNContext,
    pool_params: &Config,
    input_dims: &Dims,
    src: &[i16],
    filter_dims: &Dims,
    output_dims: &Dims,
    dst: &mut [i16],
) -> Result<()> {
    test_length!(
        src,
        input_dims.0.n * input_dims.0.h * input_dims.0.w * input_dims.0.c
    )?;
    test_length!(
        dst,
        output_dims.0.n * output_dims.0.h * output_dims.0.w * output_dims.0.c
    )?;

    let status = unsafe {
        crate::private::arm_avgpool_s16(
            ctx.as_ref(),
            pool_params.as_ref(),
            input_dims.as_ref(),
            src.as_ptr() as *const _,
            filter_dims.as_ref(),
            output_dims.as_ref(),
            dst.as_mut_ptr() as *mut _,
        )
    };

    status.check_status()
}

pub fn avgpool_s8_get_buffer_size(output_x: i32, ch_src: i32) -> i32 {
    unsafe { crate::private::arm_avgpool_s8_get_buffer_size(output_x, ch_src) }
}

pub fn avgpool_s16_get_buffer_size(output_x: i32, ch_src: i32) -> i32 {
    unsafe { crate::private::arm_avgpool_s16_get_buffer_size(output_x, ch_src) }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    const REPEAT_NUM: usize = 2;

    const MAXPOOLING_BATCH_SIZE: i32 = 2;
    const MAXPOOLING_INPUT_W: i32 = 22;
    const MAXPOOLING_INPUT_H: i32 = 12;
    const MAXPOOLING_INPUT_C: i32 = 8;
    const MAXPOOLING_FILTER_W: i32 = 6;
    const MAXPOOLING_FILTER_H: i32 = 5;
    const MAXPOOLING_STRIDE_W: i32 = 9;
    const MAXPOOLING_STRIDE_H: i32 = 5;
    const MAXPOOLING_ACTIVATION_MAX: i32 = 127;
    const MAXPOOLING_ACTIVATION_MIN: i32 = -128;
    const MAXPOOLING_OUTPUT_C: i32 = 8;
    const MAXPOOLING_OUTPUT_W: i32 = 3;
    const MAXPOOLING_OUTPUT_H: i32 = 3;
    const MAXPOOLING_PADDING_H: i32 = 1;
    const MAXPOOLING_PADDING_W: i32 = 1;

    pub fn maxpooling_arm_max_pool_s8() {
        let mut output = [0i8; (MAXPOOLING_BATCH_SIZE
            * MAXPOOLING_OUTPUT_W
            * MAXPOOLING_OUTPUT_H
            * MAXPOOLING_OUTPUT_C) as usize];
        let input_data = &crate::test_data::maxpooling_input_tensor;

        let ctx = NNContext::default();
        let pool_params = Config::new(
            (MAXPOOLING_STRIDE_W, MAXPOOLING_STRIDE_H),
            (MAXPOOLING_PADDING_W, MAXPOOLING_PADDING_H),
            (MAXPOOLING_ACTIVATION_MIN, MAXPOOLING_ACTIVATION_MAX),
        );
        let input_dims = Dims::new(
            MAXPOOLING_BATCH_SIZE,
            MAXPOOLING_INPUT_H,
            MAXPOOLING_INPUT_W,
            MAXPOOLING_INPUT_C,
        );
        let filter_dims = Dims::new(0, MAXPOOLING_FILTER_H, MAXPOOLING_FILTER_W, 0);
        let output_dims = Dims::new(
            MAXPOOLING_BATCH_SIZE,
            MAXPOOLING_OUTPUT_H,
            MAXPOOLING_OUTPUT_W,
            MAXPOOLING_OUTPUT_C,
        );

        for _ in 0..REPEAT_NUM {
            assert!(max_pool_s8(
                &ctx,
                &pool_params,
                &input_dims,
                input_data,
                &filter_dims,
                &output_dims,
                &mut output
            )
            .is_ok());
        }
    }

    pub fn maxpooling_1_arm_max_pool_s8() {}

    pub fn maxpooling_2_arm_max_pool_s8() {}

    pub fn maxpooling_3_arm_max_pool_s8() {}

    pub fn maxpooling_4_arm_max_pool_s8() {}

    pub fn maxpooling_5_arm_max_pool_s8() {}

    pub fn maxpooling_6_arm_max_pool_s8() {}

    pub fn maxpooling_7_arm_max_pool_s8() {}

    pub fn maxpooling_param_fail_arm_max_pool_s8() {}
}
