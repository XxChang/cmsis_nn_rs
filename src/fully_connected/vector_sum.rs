use crate::{Result, StatusCode};

pub fn vector_sum_s8(
    vector_sum_buf: &mut [i32],
    vector_cols: i32,
    vector_rows: i32,
    vector_data: &[i8],
    lhs_offset: i32,
    rhs_offset: i32,
    bias_data: &[i32],
) -> Result<()> {
    unsafe {
        crate::private::arm_vector_sum_s8(
            vector_sum_buf.as_mut_ptr(),
            vector_cols,
            vector_rows,
            vector_data.as_ptr(),
            lhs_offset,
            rhs_offset,
            bias_data.as_ptr(),
        )
    }
    .check_status()
}

pub fn vector_sum_s8_s64(
    vector_sum_buf: &mut [i64],
    vector_cols: i32,
    vector_rows: i32,
    vector_data: &[i8],
    lhs_offset: i32,
    bias_data: &[i64],
) -> Result<()> {
    unsafe {
        crate::private::arm_vector_sum_s8_s64(
            vector_sum_buf.as_mut_ptr(),
            vector_cols,
            vector_rows,
            vector_data.as_ptr(),
            lhs_offset,
            bias_data.as_ptr(),
        )
    }
    .check_status()
}
