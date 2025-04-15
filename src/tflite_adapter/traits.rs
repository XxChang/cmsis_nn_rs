use core::any::TypeId;

pub trait TensorOp {
    // tensor.dims() can return a empty array in the
    // case of a scalar tensor.
    fn dims(&self) -> &[i32];

    fn data_typ(&self) -> TypeId;

    fn is_data_stored_in(&self) -> bool;

    fn get_data_i32(&self) -> &[i32] {
        unimplemented!()
    }

    fn get_data_i8(&self) -> &[i8] {
        unimplemented!()
    }

    fn zero_points(&self) -> &[i32];

    fn scales(&self) -> &[f32];
}

pub trait FromLeBytes<const Bytes: usize> {
    fn create_from_le_bytes(bytes: [u8; Bytes]) -> Self;
}

impl FromLeBytes<4> for i32 {
    fn create_from_le_bytes(bytes: [u8; 4]) -> Self {
        i32::from_le_bytes(bytes)
    }
}

impl FromLeBytes<1> for i8 {
    fn create_from_le_bytes(bytes: [u8; 1]) -> Self {
        bytes[0] as i8
    }
}
