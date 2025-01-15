use crate::Result;

// s8 version of concatenation along the W axis
pub fn concatenation_s8_w(
    input: &[i8],
    input_x: u16,
    input_y: u16,
    input_z: u16,
    input_w: u16,
    output: &mut [i8],
    offset_w: u32,
) -> Result<()> {
    unsafe {
        crate::private::arm_concatenation_s8_w(
            input.as_ptr(),
            input_x,
            input_y,
            input_z,
            input_w,
            output.as_mut_ptr(),
            offset_w,
        )
    };

    Ok(())
}

// s8 version of concatenation along the X axis
pub fn concatenation_s8_x(
    input: &[i8],
    input_x: u16,
    input_y: u16,
    input_z: u16,
    input_w: u16,
    output: &mut [i8],
    output_x: u16,
    offset_x: u32,
) -> Result<()> {
    unsafe {
        crate::private::arm_concatenation_s8_x(
            input.as_ptr(),
            input_x,
            input_y,
            input_z,
            input_w,
            output.as_mut_ptr(),
            output_x,
            offset_x,
        )
    };

    Ok(())
}

// s8 version of concatenation along the Y axis
pub fn concatenation_s8_y(
    input: &[i8],
    input_x: u16,
    input_y: u16,
    input_z: u16,
    input_w: u16,
    output: &mut [i8],
    output_y: u16,
    offset_y: u32,
) -> Result<()> {
    unsafe {
        crate::private::arm_concatenation_s8_y(
            input.as_ptr(),
            input_x,
            input_y,
            input_z,
            input_w,
            output.as_mut_ptr(),
            output_y,
            offset_y,
        )
    };

    Ok(())
}

// s8 version of concatenation along the X axis
pub fn concatenation_s8_z(
    input: &[i8],
    input_x: u16,
    input_y: u16,
    input_z: u16,
    input_w: u16,
    output: &mut [i8],
    output_z: u16,
    offset_z: u32,
) -> Result<()> {
    unsafe {
        crate::private::arm_concatenation_s8_z(
            input.as_ptr(),
            input_x,
            input_y,
            input_z,
            input_w,
            output.as_mut_ptr(),
            output_z,
            offset_z,
        )
    };

    Ok(())
}
