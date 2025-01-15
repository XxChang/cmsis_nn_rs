use crate::{Result, StatusCode};

pub fn pad_s8(
    input: &[i8],
    output: &mut [i8],
    pad_value: i8,
    input_size: &crate::Dims,
    pre_pad: &crate::Dims,
    post_pad: &crate::Dims,
) -> Result<()> {
    unsafe {
        crate::private::arm_pad_s8(
            input.as_ptr(),
            output.as_mut_ptr(),
            pad_value,
            input_size.as_ref(),
            pre_pad.as_ref(),
            post_pad.as_ref(),
        )
    }
    .check_status()
}
