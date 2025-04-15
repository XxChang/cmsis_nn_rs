use super::{
    alloc, schema_generated,
    traits::{FromLeBytes, TensorOp},
};
use alloc::vec::Vec;

pub struct Tensor<const Bytes: usize, T: FromLeBytes<Bytes>> {
    data: Vec<T>,
    dims: Vec<i32>,
    zero_points: Vec<i32>,
    scales: Vec<f32>,
}

impl<const Bytes: usize, T: FromLeBytes<Bytes>> Tensor<Bytes, T> {
    pub fn create_frome_tflite(tensor: &schema_generated::tflite::Tensor<'_>, data: &[u8]) -> Self {
        let shape: Vec<i32> = tensor.shape().iter().flatten().map(|x| x).collect();

        let data: Vec<T> = data
            .chunks_exact(Bytes)
            .map(|x| {
                let mut tmp = [0; Bytes];
                tmp.copy_from_slice(x);
                T::create_from_le_bytes(tmp)
            })
            .collect();

        let mut scales = Vec::new();
        let mut zero_points = Vec::new();
        if let Some(src_quantization) = tensor.quantization() {
            if let Some((scale, zero_point)) =
                src_quantization.scale().zip(src_quantization.zero_point())
            {
                scales = scale.iter().map(|x| x).collect();
                zero_points = zero_point.iter().map(|x| x as i32).collect();
            }
        }

        Self {
            data,
            dims: shape,
            scales,
            zero_points,
        }
    }
}

pub type TensorInt8 = Tensor<1, i8>;
impl TensorOp for TensorInt8 {
    fn dims(&self) -> &[i32] {
        &self.dims
    }

    fn data_typ(&self) -> core::any::TypeId {
        core::any::TypeId::of::<i8>()
    }

    fn is_data_stored_in(&self) -> bool {
        !self.data.is_empty()
    }

    fn get_data_i8(&self) -> &[i8] {
        &self.data
    }

    fn zero_points(&self) -> &[i32] {
        &self.zero_points
    }

    fn scales(&self) -> &[f32] {
        &self.scales
    }
}

pub type TensorInt32 = Tensor<4, i32>;
impl TensorOp for TensorInt32 {
    fn dims(&self) -> &[i32] {
        &self.dims
    }

    fn data_typ(&self) -> core::any::TypeId {
        core::any::TypeId::of::<i32>()
    }

    fn is_data_stored_in(&self) -> bool {
        !self.data.is_empty()
    }

    fn get_data_i32(&self) -> &[i32] {
        &self.data
    }

    fn zero_points(&self) -> &[i32] {
        &self.zero_points
    }

    fn scales(&self) -> &[f32] {
        &self.scales
    }
}
