use crate::{private::cmsis_nn_bmm_params, NNContext, Result, StatusCode};

pub struct Config(cmsis_nn_bmm_params);

impl Config {
    pub fn new(adj_x: bool, adj_y: bool, fc_params: &super::FcParams) -> Config {
        Config(cmsis_nn_bmm_params {
            adj_x,
            adj_y,
            fc_params: fc_params.0,
        })
    }
}

impl AsRef<cmsis_nn_bmm_params> for Config {
    fn as_ref(&self) -> &cmsis_nn_bmm_params {
        &self.0
    }
}

pub fn batch_mat_mul_s8(
    ctx: &NNContext,
    bmm_params: &Config,
    quant_params: &crate::PerTensorQuantParams,
    input_lhs_dims: &crate::Dims,
    input_lhs: &[i8],
    input_rhs_dims: &crate::Dims,
    input_rhs: &[i8],
    output_dims: &crate::Dims,
    output: &mut [i8],
) -> Result<()> {
    let status = unsafe {
        crate::private::arm_batch_matmul_s8(
            ctx.as_ref(),
            bmm_params.as_ref(),
            quant_params.as_ref(),
            input_lhs_dims.as_ref(),
            input_lhs.as_ptr(),
            input_rhs_dims.as_ref(),
            input_rhs.as_ptr(),
            output_dims.as_ref(),
            output.as_mut_ptr(),
        )
    };

    status.check_status()
}

pub fn batch_mat_mul_s16(
    ctx: &NNContext,
    bmm_params: &Config,
    quant_params: &crate::PerTensorQuantParams,
    input_lhs_dims: &crate::Dims,
    input_lhs: &[i16],
    input_rhs_dims: &crate::Dims,
    input_rhs: &[i16],
    output_dims: &crate::Dims,
    output: &mut [i16],
) -> Result<()> {
    let status = unsafe {
        crate::private::arm_batch_matmul_s16(
            ctx.as_ref(),
            bmm_params.as_ref(),
            quant_params.as_ref(),
            input_lhs_dims.as_ref(),
            input_lhs.as_ptr(),
            input_rhs_dims.as_ref(),
            input_rhs.as_ptr(),
            output_dims.as_ref(),
            output.as_mut_ptr(),
        )
    };

    status.check_status()
}
