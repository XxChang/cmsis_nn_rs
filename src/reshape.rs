use crate::test_length;
use crate::{Error, Result};

pub fn reshape_s8(input: &[i8], output: &mut [i8], total_size: u32) -> Result<()> {
    test_length!(input, output)?;

    unsafe {
        crate::private::arm_reshape_s8(input.as_ptr(), output.as_mut_ptr(), total_size);
    };

    Ok(())
}
