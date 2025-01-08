use crate::private::{arm_nn_activation_s16, arm_nn_activation_type_ARM_SIGMOID, arm_nn_activation_type_ARM_TANH, arm_relu6_s8, arm_relu_q15, arm_relu_q7};
use crate::{test_length, Error, Result, StatusCode};

pub fn relu_q7(data: &mut [i8]) -> Result<()> {
    let block_size = data.len().try_into().map_err(|_| Error::Argument)?;

    unsafe { arm_relu_q7(data.as_mut_ptr(), block_size) };

    Ok(())
}

pub fn relu_q15(data: &mut [i16]) -> Result<()> {
    let block_size = data.len().try_into().map_err(|_| Error::Argument)?;

    unsafe { arm_relu_q15(data.as_mut_ptr(), block_size) };

    Ok(())
}

pub fn relu6_s8(data: &mut [i8]) -> Result<()> {
    let block_size = data.len().try_into().map_err(|_| Error::Argument)?;

    unsafe { arm_relu6_s8(data.as_mut_ptr(), block_size) };

    Ok(())
}

pub fn tanh_s16(input: &[i16], output: &mut [i16], left_shift: i32) -> Result<()> {
    test_length!(input, output)?;
    let block_size = input.len().try_into().map_err(|_| Error::Argument)?;

    unsafe {
        arm_nn_activation_s16(input.as_ptr(), output.as_mut_ptr(), block_size, left_shift, arm_nn_activation_type_ARM_TANH)
    }.check_status()
}

pub fn sigmoid_s16(input: &[i16], output: &mut [i16], left_shift: i32) -> Result<()> {
    test_length!(input, output)?;
    let block_size = input.len().try_into().map_err(|_| Error::Argument)?;

    unsafe {
        arm_nn_activation_s16(input.as_ptr(), output.as_mut_ptr(), block_size, left_shift, arm_nn_activation_type_ARM_SIGMOID)
    }.check_status()
}
