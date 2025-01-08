use crate::{private::arm_softmax_s8, test_length, Error, Result};

pub fn softmax_s8(
    input: &[i8],
    num_rows: i32,
    row_size: i32,
    mult: i32,
    shift: i32,
    diff_min: i32,
    output: &mut [i8],
) -> Result<()> {
    test_length!(input, num_rows * row_size)?;
    test_length!(output, num_rows * row_size)?;

    unsafe {
        arm_softmax_s8(
            input.as_ptr(),
            num_rows,
            row_size,
            mult,
            shift,
            diff_min,
            output.as_mut_ptr(),
        )
    };
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::test_utils::validate;

    use super::*;

    const SOFTMAX_NUM_ROWS: i32 = 2;
    const SOFTMAX_ROW_SIZE: i32 = 5;
    const SOFTMAX_INPUT_MULT: i32 = 1077952640;
    const SOFTMAX_INPUT_LEFT_SHIFT: i32 = 19;
    const SOFTMAX_DIFF_MIN: i32 = -3968;
    const SOFTMAX_DST_SIZE: usize = 10;
    const REPEAT_NUM: usize = 2;
    const SOFTMAX_OUTPUT_REF: [i8; 10] = [-57, -70, -79, -86, -92, -94, -88, -54, -91, -56];
    const SOFTMAX_INPUT: [i8; 10] = [101, 49, 6, -34, -75, -79, -38, 120, -55, 115];

    pub fn test_softmax_s8() {
        let num_rows = SOFTMAX_NUM_ROWS;
        let row_size = SOFTMAX_ROW_SIZE;
        let mult = SOFTMAX_INPUT_MULT;
        let shift = SOFTMAX_INPUT_LEFT_SHIFT;
        let diff_min = SOFTMAX_DIFF_MIN;
        let input_data = SOFTMAX_INPUT.as_slice();
        let mut output = [0i8; SOFTMAX_DST_SIZE];
        let softmax_output_ref = SOFTMAX_OUTPUT_REF.as_slice();

        for _ in 0..REPEAT_NUM {
            softmax_s8(
                input_data,
                num_rows,
                row_size,
                mult,
                shift,
                diff_min,
                output.as_mut_slice(),
            )
            .unwrap();
            validate(
                output.as_ptr(),
                softmax_output_ref.as_ptr(),
                SOFTMAX_DST_SIZE,
            );
        }
        assert!(true)
    }
}
