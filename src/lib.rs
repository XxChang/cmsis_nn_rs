#![no_std]
#![no_main]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use core::marker::PhantomData;

pub mod activation;
pub mod basic;
pub mod convolution;
pub mod fully_connected;
pub mod pooling;
pub mod softmax;

#[allow(unused)]
mod private {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[derive(Debug)]
pub enum Error {
    Argument,
    NoImpl,
    Failure,
}

pub struct NNContext<'a> {
    context: private::cmsis_nn_context,
    _marker: PhantomData<&'a ()>,
}

impl AsRef<private::cmsis_nn_context> for NNContext<'_> {
    fn as_ref(&self) -> &private::cmsis_nn_context {
        &self.context
    }
}

pub struct Dims(pub(crate) private::cmsis_nn_dims);

pub struct PerChannelQuantParams<'a> {
    params: private::cmsis_nn_per_channel_quant_params,
    _marker: PhantomData<&'a ()>,
}

impl<'a> PerChannelQuantParams<'a> {
    pub fn new(multiplier: &'a [i32], shift: &'a [i32]) -> PerChannelQuantParams<'a> {
        PerChannelQuantParams {
            params: private::cmsis_nn_per_channel_quant_params {
                multiplier: multiplier.as_ptr() as *mut i32,
                shift: shift.as_ptr() as *mut i32,
            },
            _marker: PhantomData,
        }
    }
}

impl AsRef<private::cmsis_nn_per_channel_quant_params> for PerChannelQuantParams<'_> {
    fn as_ref(&self) -> &private::cmsis_nn_per_channel_quant_params {
        &self.params
    }
}

pub struct PerTensorQuantParams(private::cmsis_nn_per_tensor_quant_params);

impl PerTensorQuantParams {
    pub fn new(multiplier: i32, shift: i32) -> PerTensorQuantParams {
        PerTensorQuantParams(private::cmsis_nn_per_tensor_quant_params { multiplier, shift })
    }
}

impl AsRef<private::cmsis_nn_per_tensor_quant_params> for PerTensorQuantParams {
    fn as_ref(&self) -> &private::cmsis_nn_per_tensor_quant_params {
        &self.0
    }
}

pub struct QuantParams<'a> {
    params: private::cmsis_nn_quant_params,
    _marker: PhantomData<&'a ()>,
}

impl<'a> QuantParams<'a> {
    pub fn new(multiplier: &'a [i32], shift: &'a [i32], is_per_channel: i32) -> QuantParams<'a> {
        QuantParams {
            params: private::cmsis_nn_quant_params {
                multiplier: multiplier.as_ptr() as *mut i32,
                shift: shift.as_ptr() as *mut i32,
                is_per_channel,
            },
            _marker: PhantomData,
        }
    }
}

impl AsRef<private::cmsis_nn_quant_params> for QuantParams<'_> {
    fn as_ref(&self) -> &private::cmsis_nn_quant_params {
        &self.params
    }
}

impl Dims {
    pub fn new(patch_size: i32, height: i32, width: i32, channels: i32) -> Dims {
        Dims(private::cmsis_nn_dims {
            n: patch_size,
            h: height,
            w: width,
            c: channels,
        })
    }
}

impl AsRef<private::cmsis_nn_dims> for Dims {
    fn as_ref(&self) -> &private::cmsis_nn_dims {
        &self.0
    }
}

type Result<T> = core::result::Result<T, Error>;

pub trait StatusCode {
    fn check_status(self) -> Result<()>;
}

impl StatusCode for private::arm_cmsis_nn_status {
    fn check_status(self) -> Result<()> {
        match self {
            private::arm_cmsis_nn_status_ARM_CMSIS_NN_SUCCESS => Ok(()),
            private::arm_cmsis_nn_status_ARM_CMSIS_NN_ARG_ERROR => Err(Error::Argument),
            private::arm_cmsis_nn_status_ARM_CMSIS_NN_NO_IMPL_ERROR => Err(Error::NoImpl),
            _ => Err(Error::Failure),
        }
    }
}

#[macro_export]
macro_rules! test_length {
    ($input1:ident, $input2:ident) => {
        if $input1.len() == $input2.len() {
            Ok(())
        } else {
            Err(Error::Argument)
        }
    };
    ($input:ident, $len:expr) => {
        if $input.len() == $len as usize {
            Ok(())
        } else {
            Err(Error::Argument)
        }
    };
    ($input1:ident, $input2:ident, $output:ident) => {
        if $input1.len() == $input2.len() && $input1.len() == $output.len() {
            Ok(())
        } else {
            Err(Error::Argument)
        }
    };
}

impl Default for NNContext<'_> {
    fn default() -> Self {
        Self {
            context: private::cmsis_nn_context {
                buf: core::ptr::null_mut(),
                size: 0,
            },
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod test_utils {
    pub fn validate(act: *const i8, ref_data: *const i8, size: usize) -> bool {
        let mut test_passed = true;
        let mut count = 0;
        let mut total = 0;

        for i in 0..size {
            total += 1;
            let act_data = unsafe { *act.offset(i as _) };
            let ref_data = unsafe { *ref_data.offset(i as _) };
            if act_data != ref_data {
                defmt::error!("ERROR at pos {}: Act: {} Ref: {}", i, act_data, ref_data);
                count += 1;
                test_passed = false;
            }
        }

        if !test_passed {
            defmt::error!("{} of {} failed", count, total);
        }

        test_passed
    }
}

#[cfg(test)]
mod test_data;

#[cfg(test)]
#[defmt_test::tests]
mod tests {
    use defmt_rtt as _;
    use nrf52833_hal as _;
    use panic_probe as _;

    #[test]
    fn test_elementwise_add_s8() {
        crate::basic::tests::test_elementwise_add_s8();
    }

    #[test]
    fn test_softmax_s8() {
        crate::softmax::tests::test_softmax_s8();
    }

    #[test]
    fn test_arm_max_pool_s8() {
        crate::pooling::tests::maxpooling_arm_max_pool_s8();
        crate::pooling::tests::maxpooling_1_arm_max_pool_s8();
        crate::pooling::tests::maxpooling_2_arm_max_pool_s8();
        crate::pooling::tests::maxpooling_3_arm_max_pool_s8();
        crate::pooling::tests::maxpooling_4_arm_max_pool_s8();
        crate::pooling::tests::maxpooling_5_arm_max_pool_s8();
        crate::pooling::tests::maxpooling_6_arm_max_pool_s8();
        crate::pooling::tests::maxpooling_7_arm_max_pool_s8();
        crate::pooling::tests::maxpooling_param_fail_arm_max_pool_s8();
    }
}
