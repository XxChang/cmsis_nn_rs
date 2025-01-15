use crate::private::arm_elementwise_add_s16;
use crate::{private::arm_elementwise_add_s8, StatusCode};
use crate::{test_length, Dims, Error, NNContext, Result};

/// Elementwise add of two vectors.
/// <div>
/// \(a + b \)
/// </div>
pub fn elementwise_add_s8(
    input_1: &[i8],
    input_2: &[i8],
    input_1_offset: i32,
    input_1_mult: i32,
    input_1_shift: i32,
    input_2_offset: i32,
    input_2_mult: i32,
    input_2_shift: i32,
    left_shift: i32,
    output: &mut [i8],
    out_offset: i32,
    out_mult: i32,
    out_shift: i32,
    out_activation_min: i32,
    out_activation_max: i32,
) -> Result<()> {
    test_length!(input_1, input_2, output)?;

    let block_size: i32 = input_1.len().try_into().map_err(|_| Error::Argument)?;

    unsafe {
        arm_elementwise_add_s8(
            input_1.as_ptr(),
            input_2.as_ptr(),
            input_1_offset,
            input_1_mult,
            input_1_shift,
            input_2_offset,
            input_2_mult,
            input_2_shift,
            left_shift,
            output.as_mut_ptr(),
            out_offset,
            out_mult,
            out_shift,
            out_activation_min,
            out_activation_max,
            block_size,
        )
    }
    .check_status()
}

pub fn elementwise_add_s16(
    input_1: &[i16],
    input_2: &[i16],
    input_1_offset: i32,
    input_1_mult: i32,
    input_1_shift: i32,
    input_2_offset: i32,
    input_2_mult: i32,
    input_2_shift: i32,
    left_shift: i32,
    output: &mut [i16],
    out_offset: i32,
    out_mult: i32,
    out_shift: i32,
    out_activation_min: i32,
    out_activation_max: i32,
) -> Result<()> {
    test_length!(input_1, input_2, output)?;

    let block_size: i32 = input_1.len().try_into().map_err(|_| Error::Argument)?;

    unsafe {
        arm_elementwise_add_s16(
            input_1.as_ptr(),
            input_2.as_ptr(),
            input_1_offset,
            input_1_mult,
            input_1_shift,
            input_2_offset,
            input_2_mult,
            input_2_shift,
            left_shift,
            output.as_mut_ptr(),
            out_offset,
            out_mult,
            out_shift,
            out_activation_min,
            out_activation_max,
            block_size,
        )
    }
    .check_status()
}

pub fn elementwise_mul_acc_s16(
    input_1_vect: &[i16],
    input_2_vect: &[i16],
    input_1_offset: i32,
    input_2_offset: i32,
    output: &mut [i16],
    out_offset: i32,
    out_mult: i32,
    out_shift: i32,
    out_activation_min: i32,
    out_activation_max: i32,
) -> Result<()> {
    test_length!(input_1_vect, input_2_vect, output)?;

    let block_size: i32 = input_1_vect.len().try_into().map_err(|_| Error::Argument)?;

    unsafe {
        crate::private::arm_elementwise_mul_acc_s16(
            input_1_vect.as_ptr(),
            input_2_vect.as_ptr(),
            input_1_offset,
            input_2_offset,
            output.as_mut_ptr(),
            out_offset,
            out_mult,
            out_shift,
            out_activation_min,
            out_activation_max,
            block_size,
        )
    }
    .check_status()
}

pub fn elementwise_mul_s8(
    input_1_vect: &[i8],
    input_2_vect: &[i8],
    input_1_offset: i32,
    input_2_offset: i32,
    output: &mut [i8],
    out_offset: i32,
    out_mult: i32,
    out_shift: i32,
    out_activation_min: i32,
    out_activation_max: i32,
) -> Result<()> {
    test_length!(input_1_vect, input_2_vect, output)?;

    let block_size: i32 = input_1_vect.len().try_into().map_err(|_| Error::Argument)?;

    unsafe {
        crate::private::arm_elementwise_mul_s8(
            input_1_vect.as_ptr(),
            input_2_vect.as_ptr(),
            input_1_offset,
            input_2_offset,
            output.as_mut_ptr(),
            out_offset,
            out_mult,
            out_shift,
            out_activation_min,
            out_activation_max,
            block_size,
        )
    }
    .check_status()
}

// s16 element wise multiplication of batches of two vectors
pub fn elementwise_mul_s16_batch_offset(
    input_1_vect: &[i16],
    input_2_vect: &[i16],
    output: &mut [i16],
    out_offset: i32,
    out_mult: i32,
    out_shift: i32,
    block_size: i32,
    batch_size: i32,
    batch_offset: i32,
) -> Result<()> {
    test_length!(input_1_vect, input_2_vect, output)?;

    unsafe {
        crate::private::arm_elementwise_mul_s16_batch_offset(
            input_1_vect.as_ptr(),
            input_2_vect.as_ptr(),
            output.as_mut_ptr(),
            out_offset,
            out_mult,
            out_shift,
            block_size,
            batch_size,
            batch_offset,
        )
    }
    .check_status()
}

// s16 elementwise multiplication with s8 output
pub fn elementwise_mul_s16_s8(
    input_1_vect: &[i16],
    input_2_vect: &[i16],
    output: &mut [i8],
    out_offset: i32,
    out_mult: i32,
    out_shift: i32,
    block_size: i32,
    batch_size: i32,
    batch_offset: i32,
) -> Result<()> {
    test_length!(input_1_vect, input_2_vect, output)?;

    unsafe {
        crate::private::arm_elementwise_mul_s16_s8(
            input_1_vect.as_ptr(),
            input_2_vect.as_ptr(),
            output.as_mut_ptr(),
            out_offset,
            out_mult,
            out_shift,
            block_size,
            batch_size,
            batch_offset,
        )
    }
    .check_status()
}

pub fn elementwise_mul_s16(
    input_1_vect: &[i16],
    input_2_vect: &[i16],
    input_1_offset: i32,
    input_2_offset: i32,
    output: &mut [i16],
    out_offset: i32,
    out_mult: i32,
    out_shift: i32,
    out_activation_min: i32,
    out_activation_max: i32,
) -> Result<()> {
    test_length!(input_1_vect, input_2_vect, output)?;

    let block_size: i32 = input_1_vect.len().try_into().map_err(|_| Error::Argument)?;

    unsafe {
        crate::private::arm_elementwise_mul_s16(
            input_1_vect.as_ptr(),
            input_2_vect.as_ptr(),
            input_1_offset,
            input_2_offset,
            output.as_mut_ptr(),
            out_offset,
            out_mult,
            out_shift,
            out_activation_min,
            out_activation_max,
            block_size,
        )
    }
    .check_status()
}

pub fn maximum_s8(
    ctx: &NNContext,
    input_1_data: &[i8],
    input_1_dims: &Dims,
    input_2_data: &[i8],
    input_2_dims: &Dims,
    output_data: &mut [i8],
    output_dims: &Dims,
) -> Result<()> {
    test_length!(
        input_1_data,
        input_1_dims.0.n * input_1_dims.0.h * input_1_dims.0.w * input_1_dims.0.c
    )?;
    test_length!(
        input_2_data,
        input_2_dims.0.n * input_2_dims.0.h * input_2_dims.0.w * input_2_dims.0.c
    )?;
    test_length!(
        output_data,
        output_dims.0.n * output_dims.0.h * output_dims.0.w * output_dims.0.c
    )?;

    let status = unsafe {
        crate::private::arm_maximum_s8(
            ctx.as_ref(),
            input_1_data.as_ptr(),
            input_1_dims.as_ref(),
            input_2_data.as_ptr(),
            input_2_dims.as_ref(),
            output_data.as_mut_ptr(),
            output_dims.as_ref(),
        )
    };

    status.check_status()
}

pub fn minimum_s8(
    ctx: &NNContext,
    input_1_data: &[i8],
    input_1_dims: &Dims,
    input_2_data: &[i8],
    input_2_dims: &Dims,
    output_data: &mut [i8],
    output_dims: &Dims,
) -> Result<()> {
    test_length!(
        input_1_data,
        input_1_dims.0.n * input_1_dims.0.h * input_1_dims.0.w * input_1_dims.0.c
    )?;
    test_length!(
        input_2_data,
        input_2_dims.0.n * input_2_dims.0.h * input_2_dims.0.w * input_2_dims.0.c
    )?;
    test_length!(
        output_data,
        output_dims.0.n * output_dims.0.h * output_dims.0.w * output_dims.0.c
    )?;

    let status = unsafe {
        crate::private::arm_minimum_s8(
            ctx.as_ref(),
            input_1_data.as_ptr(),
            input_1_dims.as_ref(),
            input_2_data.as_ptr(),
            input_2_dims.as_ref(),
            output_data.as_mut_ptr(),
            output_dims.as_ref(),
        )
    };

    status.check_status()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::test_utils::validate;

    const ADD_DST_SIZE: usize = 128;

    const ADD_INPUT1: [i8; ADD_DST_SIZE] = [
        -92, -108, 69, -105, 33, 9, 125, -17, -120, 8, 109, -52, 68, -113, -97, 92, 73, -58, 72,
        56, -28, -106, -56, 101, -50, -5, -89, -24, 54, -31, 15, 117, -68, 37, 72, 41, -68, 113,
        -10, 83, 3, 62, 125, 57, 124, -119, -65, 93, -122, 120, 105, -87, 101, -75, 64, -88, -62,
        122, -20, 64, -41, -51, -98, 25, 3, 84, 14, -33, 111, -37, -74, -11, 34, 121, 35, 30, -41,
        -77, -44, -109, -11, 39, 31, 60, 121, 41, -49, 50, 38, -20, -123, 33, 102, 80, 126, 111,
        -7, 41, 16, 120, 65, -67, 10, -88, -113, 107, 68, -120, -71, -102, 105, 99, -94, 125, -119,
        98, -47, 46, -22, -125, -101, 99, -20, 4, -1, -1, 65, -108,
    ];

    const ADD_INPUT2: [i8; ADD_DST_SIZE] = [
        72, 77, -78, -32, -32, 45, 108, -49, 121, 43, -25, -47, 35, 14, 87, 31, 27, 96, -15, 76,
        -31, 77, 108, 114, 90, 23, 33, 109, -72, 12, 119, -44, -19, -55, -72, 94, -22, 65, 103, 43,
        -90, -71, 5, 115, -5, -57, 59, 70, 52, 78, 73, 41, 104, -61, 89, 44, 37, 77, 91, -2, 7,
        -115, 49, 121, -49, 15, 76, -121, 100, 11, -39, -45, 86, -35, 120, -65, -9, -127, 28, 111,
        26, 104, 116, 89, 30, 97, -13, -124, -101, 94, 36, -86, -75, 49, 6, 40, 95, 99, -77, -62,
        -74, -88, -124, 97, -38, 122, -5, 3, 63, -94, 21, 18, 114, -69, 20, 19, -121, 0, -59, -65,
        -69, -71, -23, 59, 101, -55, -86, -68,
    ];

    const ADD_OUTPUT_REF: [i8; 128] = [
        -10, -15, -4, -68, 1, 27, 117, -33, 1, 26, 42, -49, 52, -49, -5, 62, 50, 19, 29, 66, -29,
        -14, 26, 108, 20, 9, -28, 43, -9, -9, 67, 37, -43, -9, 0, 68, -45, 89, 47, 63, -43, -4, 65,
        86, 60, -88, -3, 82, -35, 99, 89, -23, 103, -68, 77, -22, -12, 100, 36, 31, -17, -83, -24,
        73, -23, 50, 45, -77, 106, -13, -56, -28, 60, 43, 78, -17, -25, -102, -8, 1, 8, 72, 74, 75,
        76, 69, -31, -37, -31, 37, -43, -26, 14, 65, 66, 76, 44, 70, -30, 29, -4, -77, -57, 5, -75,
        115, 32, -58, -4, -98, 63, 59, 10, 28, -49, 59, -84, 23, -40, -95, -85, 14, -21, 32, 50,
        -28, -10, -88,
    ];

    const ADD_OUT_ACTIVATION_MIN: i32 = -128;
    const ADD_OUT_ACTIVATION_MAX: i32 = 127;
    const ADD_INPUT1_OFFSET: i32 = 128;
    const ADD_INPUT2_OFFSET: i32 = 128;
    const ADD_OUTPUT_MULT: i32 = 1073741824;
    const ADD_OUTPUT_SHIFT: i32 = -19;
    const ADD_OUTPUT_OFFSET: i32 = -128;
    const ADD_LEFT_SHIFT: i32 = 20;
    const ADD_INPUT1_SHIFT: i32 = 0;
    const ADD_INPUT2_SHIFT: i32 = 0;
    const ADD_INPUT1_MULT: i32 = 1073741824;
    const ADD_INPUT2_MULT: i32 = 1073741824;

    pub fn test_elementwise_add_s8() {
        let mut output = [0i8; ADD_DST_SIZE];
        let add_output_ref = ADD_OUTPUT_REF.as_slice();

        let input_data1 = ADD_INPUT1.as_slice();
        let input_data2 = ADD_INPUT2.as_slice();

        let input_1_mult = ADD_INPUT1_MULT;
        let input_1_shift = ADD_INPUT1_SHIFT;
        let input_1_offset = ADD_INPUT1_OFFSET;
        let input_2_mult = ADD_INPUT2_MULT;
        let input_2_shift = ADD_INPUT2_SHIFT;
        let input_2_offset = ADD_INPUT2_OFFSET;

        let left_shift = ADD_LEFT_SHIFT;

        let out_offset = ADD_OUTPUT_OFFSET;
        let out_mult = ADD_OUTPUT_MULT;
        let out_shift = ADD_OUTPUT_SHIFT;

        let out_activation_min = ADD_OUT_ACTIVATION_MIN;
        let out_activation_max = ADD_OUT_ACTIVATION_MAX;

        assert!(elementwise_add_s8(
            input_data1,
            input_data2,
            input_1_offset,
            input_1_mult,
            input_1_shift,
            input_2_offset,
            input_2_mult,
            input_2_shift,
            left_shift,
            &mut output,
            out_offset,
            out_mult,
            out_shift,
            out_activation_min,
            out_activation_max
        )
        .is_ok());

        assert!(validate(
            output.as_ptr(),
            add_output_ref.as_ptr(),
            ADD_DST_SIZE
        ));
    }
}
